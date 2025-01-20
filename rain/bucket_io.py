# -*- coding: utf-8 -*-
# IO for Page-ED
from __future__ import absolute_import
import logging
import numpy
import sys, os
import time
import copy
import math
import tqdm
import pickle
from PIL import Image
import cv2
import random
from multiprocessing import Manager, Process
from six.moves import queue
from datetime import datetime
logger = logging.getLogger()

from torch.utils.data.dataset import Dataset
import torch.distributed as dist

from . import xconfig
from .utils import read_image, load_det_sections, parse_title
from data_encapsulation import ListRecordLoader, load_annotation, count_record_info
from . import data_aug

from .RFL_rain import cs_main


def filt_noise(annotation):
    noise_topics = list()
    valid_topics = list()
    for topic in annotation.topics:
        noise_columns = list()
        valid_columns = list()
        for column in topic.columns:
            noise_lines = list()
            valid_lines = list()
            for line in column.lines:
                if line.idx == '':
                    noise_lines.append(line)
                else:
                    valid_lines.append(line)
            column.ignore_lines.extend(noise_lines)
            column.lines = valid_lines
            if (column.idx == '') or (len(valid_lines) == 0):
                noise_columns.append(column)
            else:
                valid_columns.append(column)
        topic.ignore_columns.extend(noise_columns)
        topic.columns = valid_columns
        if (topic.idx == '') or (len(topic.columns) == 0):
            noise_topics.append(topic)
        else:
            valid_topics.append(topic)
    annotation.ignore_topics.extend(noise_topics)
    annotation.topics = valid_topics


def info_count_func_worker(record_loader, level, indices, keys_queue=None, records_queue=None):
    assert level in ['topic', 'column', 'line']

    def cal_w_h(segmentation):
        x1, y1, x2, y2 = segmentation.bbox
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        x2 = math.ceil(x2)
        y2 = math.ceil(y2)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        return w, h

    keys = list()
    for i in indices:
        record = record_loader.get_record(i)
        annotation = load_annotation(record)
        annotation.crop((0, 0, annotation.image_size[0] - 1, annotation.image_size[1] - 1))
        annotation.filt_dst_ignore(recursive=True, dst_type_dict=xconfig.dst_type_dict)
        annotation.filt_overlap(recursive=True)
        filt_noise(annotation)
        annotation.remap_idx(recursive=True)

        for topic in annotation.topics:
            if level == 'topic':
                w, h = cal_w_h(topic.foreground())
                keys.append(["%s-%s" % (i, topic.id), w, h, len(topic.words()), topic.char_height()])
            else:
                for column in topic.columns:
                    if level == 'column':
                        w, h = cal_w_h(column.foreground())
                        keys.append(["%s-%s-%s" % (i, topic.id, column.id), w, h, len(column.words()), column.char_height()])
                    else:
                        for line in column.lines:
                            w, h = cal_w_h(line.segmentation)
                            keys.append(["%s-%s-%s-%s" % (i, topic.id, column.id, line.id), w, h, len(line.words), line.char_height])
        if records_queue is not None:
            records_queue.put(1)
    if keys_queue is not None:
        keys_queue.put(keys)
    return keys


def mp_info_count_func(record_loader, level, num_workers=40):
    assert level in ['topic', 'column', 'line']
    indices = range(len(record_loader))
    keys = list()
    with Manager() as manager:
        keys_queue = manager.Queue()
        records_queue = manager.Queue()
        avg_count = len(indices) // num_workers
        start = 0
        end = 0
        workers = []
        for i in range(num_workers):
            end += avg_count
            if i + 1 == num_workers:
                end = len(indices)
            logger.info("Process %d: %d-%d" % (i, start, end))
            w = Process(target=info_count_func_worker, args=(record_loader, level, indices[start:end], keys_queue, records_queue))
            w.daemon = True
            w.start()
            workers.append(w)
            start = end
        tq = tqdm.tqdm(total=len(indices))
        count = 0
        while count < len(indices):
            try:
                c = records_queue.get_nowait()
            except queue.Empty:
                continue
            count += 1
            tq.update(1)
        tq.close()
        tq = tqdm.tqdm(total=num_workers)
        count = 0
        while count < num_workers:
            try:
                keys.extend(keys_queue.get())
            except queue.Empty:
                continue
            count += 1
            tq.update(1)
        tq.close()
    return keys


def info_count_func(record_loader, level):
    assert level in ['topic', 'column', 'line']

    def cal_w_h(segmentation):
        x1, y1, x2, y2 = segmentation.bbox
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        x2 = math.ceil(x2)
        y2 = math.ceil(y2)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        return w, h

    keys = list()
    for i in tqdm.tqdm(range(len(record_loader))):
        record = record_loader.get_record(i)
        annotation = load_annotation(record)
        annotation.crop((0, 0, annotation.image_size[0] - 1, annotation.image_size[1] - 1))
        annotation.filt_dst_ignore(recursive=True, dst_type_dict=xconfig.dst_type_dict)
        annotation.filt_overlap(recursive=True)
        filt_noise(annotation)
        annotation.remap_idx(recursive=True)

        for topic in annotation.topics:
            if level == 'topic':
                w, h = cal_w_h(topic.foreground())
                keys.append(["%s-%s" % (i, topic.id), w, h, len(topic.words()), topic.char_height()])
            else:
                for column in topic.columns:
                    if level == 'column':
                        w, h = cal_w_h(column.foreground())
                        keys.append(["%s-%s-%s" % (i, topic.id, column.id), w, h, len(column.words()), column.char_height()])
                    else:
                        for line in column.lines:
                            w, h = cal_w_h(line.segmentation)
                            keys.append(["%s-%s-%s-%s" % (i, topic.id, column.id, line.id), w, h, len(line.words), line.char_height])
    return keys


class LRCParser(object):
    def __init__(self, listfile, normh=40, keyfile=None, do_test=True, ignorefile=None):
        self._normh = normh
        self._keyfile = keyfile
        self._do_test = do_test
        self._ignorefile = ignorefile
        self.record_loader = ListRecordLoader(listfile)
        self.parse_keys()

    def parse_keys(self):
        if not os.path.isfile(self._keyfile):
            cache_path = os.path.join(os.path.dirname(self._keyfile), '%s.%s.cache_info' % (self.record_loader._load_path.replace('/', '_').replace('\\', '_'), xconfig.level))
            cache_path = self._keyfile
            print(cache_path)
            infos = count_record_info(
                count_func=mp_info_count_func,
                record_loader=self.record_loader,
                extra_args=(xconfig.level, ),
                rely_files=('./data_encapsulation/data_package', './data_encapsulation/data_structure'),
                cache_path=cache_path
            )
        else:
            info = pickle.load(open(self._keyfile, 'rb'))
            infos = info['record_info']

        ignore_key_dict = dict()
        for key, w, h, l, ch in infos:
            ignore_key_dict[key] = False

        if (self._ignorefile is not None) and (xconfig.level == 'column'):
            with open(self._ignorefile, 'r') as F:
                for line in F:
                    ignore_key_dict[line.strip()] = True

        keys = list()
        ignore_col_num = 0
        for key, w, h, l, ch in infos:

            if ignore_key_dict[key]:
                ignore_col_num += 1
                continue
            record = self.record_loader.get_record(int(key.split("-")[0]))
            
            
            scale_ratio = float(self._normh / ch)
            if not self._do_test and xconfig.rand_resize:
                scale_ratio *= (random.random() * (xconfig.rand_resize_ratio[1] - xconfig.rand_resize_ratio[0]) + xconfig.rand_resize_ratio[0])
            w = round(w * scale_ratio)
            h = round(h * scale_ratio)
            if self._do_test:
                scale_ratio_clip = 1
                if h > xconfig.test_max_height and w > xconfig.test_max_width:
                    if h > w:
                        scale_ratio_clip = xconfig.test_max_height / float(h)
                    else:
                        scale_ratio_clip = xconfig.test_max_width / float(w)
                elif h > xconfig.test_max_height:
                    scale_ratio_clip = xconfig.test_max_height / float(h)
                elif w > xconfig.test_max_width:
                    scale_ratio_clip = xconfig.test_max_width / float(w)
                else:
                    scale_ratio_clip = 1
                w = round(w * scale_ratio_clip)
                h = round(h * scale_ratio_clip)
                keys.append((key, w, h, l, 0, 0, scale_ratio * scale_ratio_clip))
            elif xconfig.max_height is not None and xconfig.max_width is not None:
                scale_ratio_clip = 1
                if h > xconfig.max_height and w > xconfig.max_width:
                    if h > w:
                        scale_ratio_clip = xconfig.max_height / float(h)
                    else:
                        scale_ratio_clip = xconfig.max_width / float(w)
                elif h > xconfig.max_height:
                    scale_ratio_clip = xconfig.max_height / float(h)
                elif w > xconfig.max_width:
                    scale_ratio_clip = xconfig.max_width / float(w)
                else:
                    scale_ratio_clip = 1
                w = round(w * scale_ratio_clip)
                h = round(h * scale_ratio_clip)
                keys.append((key, w, h, l, 0, 0, scale_ratio * scale_ratio_clip))
            else:
                keys.append((key, w, h, l, 0, 0, scale_ratio))
            
            # if len(keys)==30:
            #     break
        #pdb.set_trace()
        if ignore_col_num > 0:
            logger.info('ignore abnormal key according to %s, total %d col' % (self._ignorefile, ignore_col_num))
        self._keys = keys
    
    def get_keys(self):
        return self._keys

    def rand_crop(self, img_shape, inside_bbox, outsize_bbox, rand_num=12, time_out=50):
        # x1, x2, y1, y2
        yy_max, xx_max, _ = img_shape

        crop_bbox = list()

        min_hw = min(outsize_bbox[1] - outsize_bbox[0], outsize_bbox[3] - outsize_bbox[2])
        if (min_hw // 2) < rand_num:
            rand_num = min_hw // 2
        for x_in, x_out in zip(inside_bbox[:2], outsize_bbox[:2]):
            x = -1
            x_s = min(x_in, x_out)
            x_l = max(x_in, x_out)

            x_time = 0
            while x < 0 or x >= xx_max:
                x = random.randint(x_s - rand_num, x_l + rand_num)
                x_time += 1

                if x_time >= time_out:
                    return outsize_bbox

            crop_bbox.append(x)

        for y_in, y_out in zip(inside_bbox[2:], outsize_bbox[2:]):
            y = -1
            y_s = min(y_in, y_out)
            y_l = max(y_in, y_out)

            y_time = 0
            while y < 0 or y >= yy_max:
                y = random.randint(y_s - rand_num, y_l + rand_num)
                y_time += 1

                if y_time >= time_out:
                    return outsize_bbox

            crop_bbox.append(y)

        # import pdb; pdb.set_trace()
        return crop_bbox

    def get_data(self, key, hw_limit=None, hw_est=None, scale=1.0):
        record_idx, *idxes = [int(item) for item in key.split('-')]
        record = self.record_loader.get_record(record_idx)
        annotation = load_annotation(record)
        
        if xconfig.level == 'topic':
            annotation_element = annotation.topics[idxes[0]]
        elif xconfig.level == 'column':
            annotation_element = annotation.topics[idxes[0]].columns[idxes[1]]
        elif xconfig.level == 'line':
            annotation_element = annotation.topics[idxes[0]].columns[idxes[1]].lines[idxes[2]]
        else:
            raise NotImplementedError()
        #pdb.set_trace()
        # if "struct_text" in annotation_element.raw_info:
        #     text_units = pickle.loads(annotation_element.raw_info["struct_text"])
        # else:
        #     text_units = ["\\smear"]
        # 转化为CS_String
        try:
            chemfig_string = " ".join(annotation_element.words)
            success, cs_tgt, branch_info_tgt, ring_branch_info_tgt, cond_data = cs_main(chemfig_string, is_show=False)
            if not success:
                return None, None, None, None, None, None, None, None, None, None
            else: # 一定要在结尾加上</s>,否则解码时不知道何时解码结束
                cs_tgt.append("</s>")
                branch_info_tgt.append(None)
                ring_branch_info_tgt.append(None)
                cond_data.append(-1)
        except BaseException as e:
            with open(xconfig.model_prefix + "_io.err.txt", "a+") as fp:
                formatted_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                fp.write("errinfo: {}  {}\n".format(formatted_time, e))
                fp.write("errrecord:{}  {}\n".format(formatted_time, " ".join(annotation_element.words)))
            return None, None, None, None, None, None, None, None, None, None
        
        annotation.filt_dst_ignore(recursive=True, dst_type_dict=xconfig.dst_type_dict)
        annotation.filt_overlap(recursive=True)
        filt_noise(annotation)
        annotation.remap_idx(recursive=True)
        #mport pdb; pdb.set_trace()
        if xconfig.level == 'line':
            foreground = annotation_element.segmentation
            char_height = annotation_element.char_height
            #words = annotation_element.words
        else:
            foreground = annotation_element.foreground()
            char_height = annotation_element.char_height()
            #words = annotation_element.words()

        x_min = 50000
        x_max = 0
        y_min = 50000
        y_max = 0
        if not self._do_test:
            lines = []
            if xconfig.level == 'line':
                lines = [annotation_element]
            elif xconfig.level == 'column':
                lines = annotation_element.lines
            else:
                for column in annotation_element.columns:
                    lines += column.lines
            for line_anno_element in lines:

                line_foreground = line_anno_element.segmentation
                x1, y1, x2, y2 = line_foreground.bbox
                x1 = math.floor(x1)
                y1 = math.floor(y1)
                x2 = math.ceil(x2)
                y2 = math.ceil(y2)

                x_min = min(x_min, x1)
                y_min = min(y_min, y1)
                x_max = max(x_max, x2)
                y_max = max(y_max, y2)

        image = numpy.array(Image.open(annotation.image_path).convert('RGB'))

        if not self._do_test and xconfig.do_random_scale_downup:
            image = data_aug.random_scale_downup(image, xconfig.random_scale_downup_range)
            
        # maskout
        if self._do_test or random.random() > xconfig.rand_bbox_rate:
            try:
                image[foreground.to_mask() == 0] = (255, 255, 255)
            except:
                image = image.transpose(1, 0, 2)
                image[foreground.to_mask() == 0] = (255, 255, 255)

        # crop image
        x1, y1, x2, y2 = foreground.bbox
        x1 = numpy.clip(x1, 0, image.shape[1])
        x2 = numpy.clip(x2, 0, image.shape[1])
        y1 = numpy.clip(y1, 0, image.shape[0])
        y2 = numpy.clip(y2, 0, image.shape[0])
        x1 = math.floor(x1)
        y1 = math.floor(y1)
        x2 = math.ceil(x2)
        y2 = math.ceil(y2)

        if x1 >= x_min:
            x_min = x1 + 1
        if x2 <= x_max:
            x_max = x2 - 1
        if y1 >= y_min:
            y_min = y1 + 1
        if y2 <= y_max:
            y_max = y2 - 1

        if not self._do_test and xconfig.rand_crop:
            xx1, xx2, yy1, yy2 = self.rand_crop(image.shape, [x_min, x_max, y_min, y_max], [x1, x2, y1, y2], rand_num=xconfig.rand_crop_pixel)
            if xx2 <= xx1:
                xx2 = xx1 + 1
            if yy2 <= yy1:
                yy2 = yy1 + 1
        else:
            yy1 = y1
            yy2 = y2 + 1
            xx1 = x1
            xx2 = x2 + 1
        
        image = image[yy1:yy2, xx1:xx2]
        # scale image
        w, h = x2 - x1 + 1, y2 - y1 + 1
        # scale = self._normh/char_height
        w = round(w * scale)
        h = round(h * scale)

        if not self._do_test and xconfig.do_blur:
            do_blur = random.random()
            if do_blur > 0.95:
                blur_flag = random.random()
                if blur_flag < 0.33:
                    image = cv2.blur(image, (5, 5))
                elif blur_flag < 0.66:
                    image = cv2.medianBlur(image, 5)
                else:
                    ii = random.randint(0, 360)
                    image = motion_blur(image, angle=ii)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        if xconfig.do_cutout:
            cutout_flage = random.random()
            if cutout_flage < xconfig.do_cut_rate:
                image = get_connected_components(image)

        # transpose image
        image = numpy.transpose(image, (2, 0, 1))
        # normalize
        image = image.astype(dtype=numpy.float32)
        image = -0.5 + (255-image) / 255
        # label
        
        unit_len = len(cs_tgt)
        memory_index = []  #[M] M:memory_size (id1, id2):id1 is index of seq, id2 is cand_angle id, 记录Superatom和Superbond
        bond_index = []
        branch_info_index = []

        memory_index_dict = {}  # 
        bond_index_dict = {}
        branch_index_dict = {}

        memory_used_mask = []  #[L, M]
        mem_update_info = -numpy.ones((unit_len), dtype=numpy.float32)
        branch_update_info = -numpy.ones((unit_len), dtype=numpy.float32) # 用来控制何时更新branch_info
        branch_target = []
        ring_branch_update_info = -numpy.ones((unit_len), dtype=numpy.float32) # 用来控制何时更新ring_branch_info
        ring_branch_target = []
        bond_update_info = -numpy.ones((unit_len), dtype=numpy.float32) # 控制何时保存Bond
        vocab = xconfig.vocab
        cur_remain_idx = []
        ori_string = copy.deepcopy(cs_tgt)
        
        for ind, unit in enumerate(cs_tgt):
            cs_tgt[ind] = [unit, None, None, None] # cs_tgt, ring_branch_info_tgt, branch_info_tgt, cond
            unit = cs_tgt[ind]
            token = unit[0]
            # 保存Super token, 保存update_info, dict等
            if '\Superatom' in token or '@' in token: # super token
                memory_index.append(ind)
                mem_update_info[ind] = len(memory_index) - 1
                memory_index_dict[ind] = len(memory_index) - 1
                cur_remain_idx.append(memory_index_dict[ind])
            # 保存bond token, 保存update_info, dict等
            if ('[:' in token and token.endswith(']')) or (token.startswith('?[') and token.endswith(']') and ',' in token): # bond_token
                bond_index.append(ind)
                bond_index_dict[ind] = len(bond_index) - 1
                bond_update_info[ind] = len(bond_index) - 1

            word_id = vocab.getID(token)  # word->vocab_id
            if word_id == 0:
                # 保存错误的token
                with open(xconfig.model_prefix + "_vocab_err.txt", "a+") as fp:
                    fp.write("oov: {}\n".format(unit[0]))
            
            # 保存branch_update_info
            if branch_info_tgt[ind] is not None and len(branch_info_tgt[ind]) > 0:
                branch_update_info[ind] = ind # 保存branch_info对应的index
            
            # 保存ring_branch_tgt, ring_branch_update_info, 都使用bond_index的索引
            if ring_branch_info_tgt[ind] is not None and len(ring_branch_info_tgt[ind]) > 0:
                branch_info_index.append(ind)
                branch_index_dict[ind] = len(branch_info_index) - 1
                ring_branch_update_info[ind] = len(branch_info_index) - 1 # 保存ring_branch_info对应的index
                for conn_tgt in ring_branch_info_tgt[ind]:
                    # branch_id = branch_index_dict[ind]
                    branch_id = branch_index_dict[ind]
                    conn_tgt = bond_index_dict[conn_tgt] # 将在cs_string中的位置映射到在所有Bond中的位置
                    ring_branch_target.append([branch_id, conn_tgt])

            unit[0] = word_id
            unit[1] = ring_branch_info_tgt[ind]
            unit[2] = branch_info_tgt[ind]
               
            # cond data在cs_tgt[3]中
            #cond tgt, 条件指导骨干字符串解码
            super_index = cond_data[ind]
            if super_index in memory_index_dict.keys():
                cond_tgt = memory_index_dict[super_index]
            else:
                cond_tgt = -1
            unit[3] = cond_tgt # 将-1替换为0
            
            memory_used_mask.append(cur_remain_idx.copy())
        # print(ori_string)
        return image, cs_tgt, memory_index, memory_used_mask, mem_update_info, branch_info_index, ring_branch_target, ring_branch_update_info, bond_index, bond_update_info


def motion_blur(image, degree=10, angle=20):
    #image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = numpy.diag(numpy.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = numpy.array(blurred, dtype=numpy.uint8)
    return blurred


def get_connected_components(imgInput):
    #imgInput = imgInput.transpose(1,2,0)
    imgInput_gray = cv2.cvtColor(imgInput, cv2.COLOR_RGB2GRAY)
    newRet, binaryThreshold = cv2.threshold(imgInput_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # dilation
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))

    rectdilation = cv2.dilate(binaryThreshold, rectkernel, iterations=1)

    outputImage_origin = imgInput.copy()
    outputImage = outputImage_origin
    #outputImage = cv2.cvtColor(outputImage_origin, cv2.COLOR_GRAY2BGR)
    npaContours, npaHierarchy = cv2.findContours(rectdilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for npaContour in npaContours:
        [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
        if cv2.contourArea(npaContour) > xconfig.min_contour_area:

            imgROI = binaryThreshold[intY:intY + intH, intX:intX + intW]

            subContours, subHierarchy = cv2.findContours(imgROI.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for subContour in subContours:
                if cv2.contourArea(subContour) >= xconfig.min_contour_area:
                    [pointX, pointY, width, height] = cv2.boundingRect(subContour)

                    rand_seed = numpy.random.uniform(0, 1)
                    # add patch
                    if width > xconfig.ignore_pixel or height > xconfig.ignore_pixel:
                        continue
                    #print(width,height)
                    if rand_seed <= xconfig.cutout_sample_rate:
                        # import pdb; pdb.set_trace()
                        mask = numpy.ones((height, width), numpy.float32)

                        if xconfig.width_rate * width < height:
                            x_length = int(width)
                            y_length = int(height / 4)

                            y = numpy.random.randint(height)

                            y1 = numpy.clip(y - y_length//2, 0, height)
                            y2 = numpy.clip(y + y_length//2, 0, height)

                            mask[y1:y2, :] = 0
                            mask = numpy.expand_dims(mask, axis=2)
                            mask = numpy.repeat(mask, 3, axis=2)

                            local_img = outputImage[intY + pointY:intY + pointY + height, intX + pointX:intX + pointX + width, :]
                            # patch_color = numpy.ones(mask[y1:y2, ].shape, numpy.float32) * 128.
                            local_img = local_img * mask
                            # local_img[y1:y2, :] += patch_color
                            outputImage[intY + pointY:intY + pointY + height, intX + pointX:intX + pointX + width, :] = local_img

                        else:
                            x_length = int(width / 3)
                            y_length = int(height)

                            y = numpy.random.randint(height)
                            x = numpy.random.randint(width)

                            x1 = numpy.clip(x - x_length//2, 0, width)
                            x2 = numpy.clip(x + x_length//2, 0, width)

                            mask[:, x1:x2] = 0.
                            mask = numpy.expand_dims(mask, axis=2)
                            mask = numpy.repeat(mask, 3, axis=2)

                            local_img = outputImage[intY + pointY:intY + pointY + height, intX + pointX:intX + pointX + width, :]
                            # patch_color = numpy.ones(mask[:, x1:x2].shape, numpy.float32) * 128.
                            local_img = local_img * mask
                            # local_img[:, x1:x2] += patch_color
                            outputImage[intY + pointY:intY + pointY + height, intX + pointX:intX + pointX + width, :] = local_img

    return outputImage


class ImgListParser(object):
    def __init__(self, listfile, normh=40, det_sections_file=None):
        self._normh = normh
        self._ids = []
        with open(listfile) as f:
            for line in f:
                self._ids.append(line.strip())
        if det_sections_file is not None:
            self.det_sections = load_det_sections(det_sections_file)
        else:
            self.det_sections = None

        self._keyfile = listfile
        # self.parse_keys()
    def parse_keys(self):
        self._keys = []
        key_info_file = self._keyfile + '_info.txt'
        if os.path.exists(key_info_file):
            logger.info('{} dose exist.'.format(key_info_file))
            with open(key_info_file, 'r') as fin:
                for line in fin:
                    
                    if len(line.strip().split()) > 3:
                        img_info, h, w, l, s, msl = line.strip().split()
                    else:
                        img_info, h, w, l, s, msl = line.strip().split()[0], 0, 0, 0, 0, 0
                    item = (img_info, int(h), int(w), int(l), int(s), int(msl), 0)
                    self._keys.append(item)
        else:
            fout = open(key_info_file, 'w')
            with open(self._keyfile, 'r') as fin:
                for line in fin:
                    line_all = line.strip()
                    imgfile = line_all
                    img_title = parse_title(imgfile)
                    if self.det_sections is not None:
                        img_segmentation = self.det_sections[img_title]
                    else:
                        img_segmentation = None
                    label = '<s> </s>'
                    img = read_image(imgfile, self._normh, img_segmentation, xconfig.img_fix_char_height)
                    if img is None:
                        logger.info('{} is None'.format(imgfile))
                        continue
                    vocab = xconfig.vocab
                    trans_label = [vocab.getID(l) for l in label.split(' ')]
                    item = (imgfile, int(img.shape[1]), int(img.shape[2]), int(len(trans_label)), 0, 0)
                    fout.write('{} {} {} {} {} {}\n'.format(imgfile, int(img.shape[1]), int(img.shape[2]), int(len(trans_label)), 0, 0))
                    self._keys.append(item)
            fout.close()

    def get_keys(self):
        return self._keys

    def get_data(self, line_all, hw_limit=None, hw_est=None, scale=None):
        imgfile = line_all
        img_title = parse_title(imgfile)
        if self.det_sections is not None:
            img_segmentation = self.det_sections[img_title]
        else:
            img_segmentation = None
        label = '<s> </s>'
        img = read_image(imgfile, self._normh, img_segmentation, xconfig.img_fix_char_height)
        vocab = xconfig.vocab
        trans_label = [vocab.getID(l) for l in label.split(' ')]
        img = numpy.asarray(img, dtype='float32')
        source = -0.5 + (255.-img) / 255.
        target = numpy.array(trans_label).astype('int32')
        return source, target


class EncDecDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_h=300,
        max_w=2000,
        max_l=60,
        fix_batch_size=24,
        max_batch_size=512,
        max_image_size=400000,
        seed=12345,
        do_shuffle=True,
        use_all=False,
        last_method='fill',
        one_key=False,
        return_name=False,
        do_test=False
    ):
        self.nrg = numpy.random.RandomState(seed)
        self._last_method = last_method  # 'ignore, build, fill'
        self._do_shuffle = do_shuffle
        self._data_parser = dataset
        self._use_all = use_all
        self._return_name = return_name
        self._encoder_down_sample = 2**len(xconfig.encoder_units)
        self._batchs = []
        self._feadim = xconfig.source_dim
        self._max_image_size = max_image_size
        self._fix_batch_size = fix_batch_size
        self._do_test = do_test

        self.max_h = max_h
        self.max_w = max_w
        self.max_l = max_l
        self.one_key = one_key
        self.max_batch_size = max_batch_size
        self.reset()

    def divide_find(self, x_list, q):

        len_key = len(x_list)
        left = 0
        right = len_key - 1

        while (left + 1) < right:
            mid = int((left+right) / 2)
            mid_value = x_list[mid]
            if q <= mid_value:
                right = mid
            else:
                left = mid

        if q <= x_list[left]:
            return left
        else:
            return right

    def calc_keys(self, max_height, max_width, max_length, one_key):
        items = self._data_parser.get_keys()
        mh = 0
        mw = 0
        ml = 0
        for item in items:
            _, h, w, l, _, _, _ = item
            if h > mh:
                mh = h
            if w > mw:
                mw = w
            if l > ml:
                ml = l

        logger.info('max_height=%d, max_width=%d, max_length=%d' % (min(max_height, mh), min(max_width, mw), min(max_length, ml)))
        #
        if not self._do_test:
            max_height = min(max_height, mh)
            max_width = min(max_width, mw)
            max_length = min(max_length, ml)

            max_height = 3800
            max_width = 3800
            max_length = 1700

            # max_height = 400
            # max_width  = 400
            # max_length = 200

        else:
            max_height = min(max_height, mh)
            max_width = min(max_width, mw)
            max_length = min(max_length, ml)

        logger.info('max_height=%d, max_width=%d, max_length=%d, %s' % (max_height, max_width, max_length, 'Test' if self._do_test else 'Train'))
        init_height = 64
        init_width = 128
        init_length = 1024

        h_step = 64
        w_step = 128
        l_step = 1024

        keys = []
        init_h = init_height if init_height < max_height else max_height
        init_w = init_width if init_width < max_width else max_width
        init_l = init_length if init_length < max_length else max_length

        h_his = []
        w_his = []
        l_his = []

        h = init_h
        while h <= max_height:
            w = init_w
            while w <= max_width:
                l = init_l
                while l <= max_length:
                    if h not in h_his:
                        h_his.append(h)
                    if w not in w_his:
                        w_his.append(w)
                    if l not in l_his:
                        l_his.append(l)
                    keys.append([h, w, l, h * w * l, 0])
                    if l < max_length and l + l_step > max_length:
                        l = max_length
                        continue
                    l += l_step
                if w < max_width and w + w_step > max_width:
                    w = max_width
                    continue
                w += w_step
            if h < max_height and h + h_step > max_height:
                h = max_height
                continue
            h += h_step
        #keys = sorted(keys,key=lambda area:area[3])

        len_key = len(keys)

        w_num = len(w_his)
        l_num = len(l_his)

        for item in items:
            _, h, w, l, s, msl, _ = item

            if h > max_height or w > max_width or l > max_length:
                continue
            h_index = self.divide_find(h_his, h)
            w_index = self.divide_find(w_his, w)
            l_index = self.divide_find(l_his, l)

            index = h_index*w_num*l_num + w_index*l_num + l_index
            keys[index][-1] += 1

        keys_ = []
        n_sample = len(items)
        if not self._do_test:
            th = n_sample * 0.001
        else:
            th = 1
        #th = n_sample * 0.001
        if self._use_all:
            th = 1
        m = 0
        # rank = 0 if not dist.is_initialized() else dist.get_rank()
        # bucket_info_path = os.path.join("./train_log/buckets_rank[{}]".format(rank))
        # fout = open(bucket_info_path, "a")
        drop_count = 0
        for key in keys:
            hh, ww, ll, _, n = key
            m += n
            if m >= th:
                keys_.append((hh, ww, ll))
                m = 0
            else:
                # if n > 0:
                #     fout.write("bucket {} drop samples num = {}\n".format(key, n))
                drop_count += n
        # fout.write("total drop count = {}\n".format(drop_count))
        # fout.close()


        if one_key:
            return [
                keys_[-1],
            ]
        return keys_

    def _make_plan(self):
        self._data_buckets = [[] for key in self._bucket_keys]
        items = self._data_parser.get_keys()
        if self._do_shuffle:
            self.nrg.shuffle(items)

        unuse_num = 0

        # h_all = []
        # w_all = []
        # l_all = []

        # for key in self._bucket_keys:
        #     _,h,w,l,_,_ = key
        #     if h not in h_all:
        #         h_all.append(h)
        #     if w not in w_all:
        #         w_all.append(w)
        #     if l not in l_all:
        #         l_all.append(l)

        for item in items:
            flag = 0
            for key, bucket in zip(self._bucket_keys, self._data_buckets):
                _, h, w, l, _, _ = key
                if item[1] <= h and item[2] <= w and item[3] <= l:
                    bucket.append(item)
                    flag = 1
                    break
            if flag == 0:
                unuse_num += 1
        logger.info("unuse_num: {}".format(unuse_num))
        logger.info("total_num: {}".format(len(items)))
        self.min_buckets = self._data_buckets[0]

        # rank = 0 if not dist.is_initialized() else dist.get_rank()
        # bucket_info_path = os.path.join("./train_log/buckets_rank[{}]".format(rank))
        # fout = open(bucket_info_path, "a")
        total_num = 0
        for key, bucket in zip(self._bucket_keys, self._data_buckets):

            if not self._do_test:
                batchsize, h, w, l, ph, pw = key

                world_size = float(dist.get_world_size())
                pad_len = int(world_size * batchsize) - int(len(bucket) % int(world_size * batchsize))
                if pad_len > 0:
                    if len(bucket) >= pad_len:
                        bucket += bucket[-pad_len:]
                    elif len(bucket) > 0:
                        bucket += pad_len * [bucket[-1]]
                assert int(len(bucket) % (world_size*batchsize)) == 0, 'len = %d, pad_len = %d, world_size = %d, batchsize = %d' % (len(bucket), pad_len, world_size, batchsize)
            if not self._do_test:
                logger.info('bucket {}, num sample: {}'.format(key, len(bucket)))
                # fout.write('bucket {}, num sample: {}\n'.format(key, len(bucket)))
                # total_num += len(bucket)
        # fout.write("total num = {}\n".format(total_num))
        # fout.write("-----------------------\n")
        # fout.close()

    def __len__(self):
        return len(self._batchs)

    def reset(self):
        self._data_parser.parse_keys()
        keys = self.calc_keys(self.max_h, self.max_w, self.max_l, self.one_key)
        self._bucket_keys = []
        mh = mw = ml = 0
        max_size = 0
        max_ind = 0
        count = 0
        for i, key in enumerate(keys):
            h, w, l = key
            mh = h if h > mh else mh
            mw = w if w > mw else mw
            ml = l if l > ml else ml
            ph = (h + self._encoder_down_sample - 1) // self._encoder_down_sample
            pw = (w + self._encoder_down_sample - 1) // self._encoder_down_sample
            if self._fix_batch_size is None:
                batchsize = int(self._max_image_size // (h*w))
                if batchsize > self.max_batch_size:
                    batchsize = self.max_batch_size
            else:
                batchsize = self._fix_batch_size
            if batchsize == 0:
                continue
            cur_size = batchsize * h * w
            if cur_size > max_size:
                max_size = cur_size
                max_ind = count
            self._bucket_keys.append((batchsize, h, w, l, ph, pw))
            count += 1

        self._make_plan()
        self._raw_bucket_keys = copy.deepcopy(self._bucket_keys)

        self._bucket_keys = copy.deepcopy(self._raw_bucket_keys)
        if self._do_shuffle:
            for bucket in self._data_buckets:
                self.nrg.shuffle(bucket)
        self._batchs = []
        extra_bucket_keys = []
        for bid, (key, bucket) in enumerate(zip(self._bucket_keys, self._data_buckets)):
            '''if bid <5:
                continue'''
            num_bucket = len(bucket)
            batchsize, h, w, l, ph, pw = key
            bnum = (num_bucket+batchsize-1) // batchsize
            for b in range(bnum):
                start = b * batchsize
                end = start + batchsize if start + batchsize < num_bucket else num_bucket

                if self._last_method == 'ignore' and end - start < batchsize:
                    continue
                if self._last_method == 'fill' and end - start < batchsize:
                    cur_batch = [bucket[start:end] + self.min_buckets[0:(batchsize - end + start)], bid]
                    if len(self.min_buckets) < batchsize - end + start:
                        left_num = batchsize - end + start - len(self.min_buckets)
                        cur_count = 0
                        for left in range(left_num):
                            cur_batch[0].append(self.min_buckets[cur_count])
                            cur_count += 1
                            if cur_count == len(self.min_buckets):
                                cur_count = 0
                    self._batchs.append(tuple(cur_batch))
                elif end - start < batchsize:
                    extra_bucket_keys.append((end - start, h, w, l, ph, pw))
                    self._batchs.append((bucket[start:end], self.num_bucket_keys + len(extra_bucket_keys) - 1))
                else:
                    self._batchs.append((bucket[start:end], bid))

        self._bucket_keys += extra_bucket_keys

        logger.info('batch num={}'.format(len(self._batchs)))

    def get_item(self, batchs, index):

        batch, bid = batchs[index]
        max_target_len = 1
        max_source_w = 16
        max_source_h = 16
        max_mem_size = 1
        max_hook_size = 1
        max_branch_size = 1
        max_bond_size = 1
        cand_angle_size = 24  # 360 = 15 * 24
        batch_list = []
        names_list = []

        w_lim, h_lim, l_lim = self._bucket_keys[bid][1:4]
        l_lim += 3
        for pos, (lkey, cw, ch, cl, _, _, scale) in enumerate(batch):
            # source, rend_units, memory_indexs, memory_remain_idx, _mem_update_info, _hook_update_info, _hook_target = self._data_parser.get_data(lkey, (h_lim, w_lim), (ch, cw), scale)

            source, cs_tgt, memory_indexs, memory_remain_idx, _mem_update_info, branch_index, ring_branch_target, _ring_branch_update_info, bond_index, bond_update_info = self._data_parser.get_data(lkey, (h_lim, w_lim), (ch, cw), scale)
            if source is None or cs_tgt is None or ring_branch_target is None:
                continue
            c, h, w = source.shape
            
            tlen = len(cs_tgt)

            mlen = len(memory_indexs) # super len
            branch_len = len(ring_branch_target)
            bond_len = len(bond_index)

            if not self._do_test:
                if abs(h - ch) > 32 or abs(w - cw) > 32:
                    print('actual:', w, h, tlen)
                    print('est:', cw, ch, cl)
                    continue
            
            batch_list.append((source, cs_tgt, memory_indexs, memory_remain_idx, _mem_update_info, branch_index, ring_branch_target, _ring_branch_update_info, bond_index, bond_update_info))
            max_target_len = max(max_target_len, tlen)
            max_source_w = max(max_source_w, w)
            max_source_h = max(max_source_h, h)
            max_mem_size = max(max_mem_size, mlen)
            # max_hook_size = max(max_hook_size, hlen)
            max_branch_size = max(max_branch_size, branch_len)
            max_bond_size = max(max_bond_size, bond_len)
            names_list.append(lkey)
  
        if max_source_w % 32 != 0:
            max_source_w += (32 - max_source_w%32)
        if max_source_h % 32 != 0:
            max_source_h += (32 - max_source_h%32)
        # 将数据转换为numpy
        batch_size = len(batch_list)
        
        data = numpy.zeros((batch_size, self._feadim, max_source_h, max_source_w), dtype='float32')
        data_mask = numpy.zeros((batch_size, 1, max_source_h, max_source_w), dtype='float32')
        target = numpy.zeros((batch_size, max_target_len), dtype='float32')
        target_mask = numpy.zeros((batch_size, max_target_len), dtype='float32')
        
        cond_data = numpy.zeros((batch_size, max_target_len), dtype='float32') # [b, l]
        mem_index_data = -numpy.ones((batch_size, max_mem_size + 1), dtype='float32')
        memory_used_mask = numpy.zeros((batch_size, max_target_len, max_mem_size + 1), dtype='float32')
        memory_used_mask[:, :, 0] = 1
        mem_update_info = -numpy.ones((batch_size, max_target_len), dtype='float32')

        ring_branch_update_info = -numpy.ones((batch_size, max_target_len), dtype='float32')
        ring_branch_target = -numpy.ones((batch_size, max_branch_size, 2), dtype='float32') # [b, l_branch, 2] # 原始的label
        ring_branch_label = numpy.zeros((batch_size, max_branch_size, max_bond_size), dtype='float32')

        bond_index_data = numpy.zeros((batch_size, max_bond_size), dtype='float32')
        bond_update_info = -numpy.ones((batch_size, max_target_len), dtype='float32')


        # 批batch,初始化data_mask, tgt_mask
        for pos, (source, cs_tgt, memory_indexs, memory_remain_idx, _mem_update_info, branch_index, branch_tgt, _branch_update_info, bond_index, _bond_update_info) in enumerate(batch_list):
            c, h, w = source.shape
            data[pos, :, :h, :w] = source
            data_mask[pos, :, :h, :w] = 1.
            tlen = len(cs_tgt)
            mlen = len(memory_indexs)
            branch_len = len(branch_tgt)
            bond_len = len(bond_index)
            for unit_id, unit in enumerate(cs_tgt):
                # token
                target[pos, unit_id] = unit[0]
                target_mask[pos, unit_id] = 1
                # cond
                cond_data[pos, unit_id] = unit[3] + 1 # 引导信息
                cur_remain_idx = memory_remain_idx[unit_id]
                for remain_idx in cur_remain_idx:
                    memory_used_mask[pos, unit_id, remain_idx + 1] = 1
            
            if mlen > 0:
                mem_index_data[pos, 1:mlen + 1] = numpy.array(memory_indexs)
                mem_update_info[pos, :tlen] = _mem_update_info + (_mem_update_info != -1)
                
            
            if branch_len > 0:
                ring_branch_update_info[pos, :tlen] = _branch_update_info
                for qid, tgt in enumerate(branch_tgt):
                    ring_branch_target[pos, qid, 0] = tgt[0] # conn begin index
                    ring_branch_target[pos, qid, 1] = tgt[1] # conn end index
                    ring_branch_label[pos, tgt[0], tgt[1]] = 1 #

            if bond_len > 0:
                bond_index_data[pos, :bond_len] = bond_index
                bond_update_info[pos, :tlen] = _bond_update_info
            
            
        # xconfig.vocab.indices2words(target[pos].tolist())
        #outputs = [data, data_mask, target, target_mask, cond_data, mem_index_data, mem_used_mask, mem_update_info]
        outputs = {}
        outputs["data"] = data
        outputs["data_mask"] = data_mask
        outputs["target"] = target
        outputs["target_mask"] = target_mask
        outputs["cond_data"] = cond_data
        outputs["mem_index_data"] = mem_index_data
        outputs["mem_used_mask"] = memory_used_mask
        outputs["mem_update_info"] = mem_update_info
        outputs["branch_target"] = ring_branch_target
        outputs["branch_label"] = ring_branch_label
        outputs["branch_update_info"] = ring_branch_update_info
        outputs["bond_index_data"] = bond_index_data
        outputs["bond_update_info"] = bond_update_info
        
        if self._return_name:
            # outputs.append(names_list)
            outputs["names_list"] = names_list
        return outputs


# add for data parallel
class DataPartioner(object):
    def __init__(self, data, size=1, rank=0, max_batch_one_epoch=40000000, seed=0):
        self.data = data
        self.partitions = []
        self.part_len = len(data) // size
        self.rank = rank
        self.max_batch_one_epoch = max_batch_one_epoch
        import random
        random.seed(seed)

        # assert int(len(self.data._batchs) % size) == 0, 'len(self.data._batchs) % world_size must be zero, %d vs 0' %(int(len(self.data._batchs) % size))
        for i in range(size):
            self.partitions.append(self.data._batchs[i::size])

            if i >= 1:
                assert (len(self.partitions[-1]) == len(self.partitions[-1]))

        random.shuffle(self.partitions[self.rank])

    def __len__(self, ):
        return len(self.partitions[self.rank])

    def __getitem__(self, index):
        index = (index+1) % len(self)
        return self.data.get_item(self.partitions[self.rank], index)


def build_data(
    lrcfile,
    keyfile=None,
    max_h=200,
    max_w=2000,
    max_l=60,
    fix_batch_size=32,
    max_batch_size=512,
    max_image_size=400000,
    seed=12345,
    do_shuffle=True,
    use_all=False,
    last_method='fill',
    one_key=False,
    return_name=False,
    ignorefile=None,
    image_list_file=None,
    normh=40,
    do_test=False
):
    if image_list_file is None:
        dataparser = LRCParser(lrcfile, normh, keyfile, do_test=do_test, ignorefile=ignorefile)
    else:
        dataparser = ImgListParser(image_list_file, normh)
    dataiter = EncDecDataset(
        dataparser,
        max_h,
        max_w,
        max_l,
        fix_batch_size=fix_batch_size,
        max_batch_size=max_batch_size,
        max_image_size=max_image_size,
        seed=seed,
        do_shuffle=do_shuffle,
        use_all=use_all,
        last_method=last_method,
        one_key=one_key,
        return_name=return_name,
        do_test=do_test
    )
    return dataiter


def build_data_test(
    max_h=200,
    max_w=2000,
    max_l=60,
    fix_batch_size=32,
    max_batch_size=512,
    max_image_size=400000,
    seed=12345,
    do_shuffle=True,
    use_all=False,
    last_method='fill',
    one_key=False,
    return_name=False,
    image_list_file=None,
    det_sections_file=None,
    normh=40,
    do_test=False
):
    dataparser = ImgListParser(image_list_file, normh, det_sections_file)
    dataiter = EncDecDataset(
        dataparser,
        max_h,
        max_w,
        max_l,
        fix_batch_size=fix_batch_size,
        max_batch_size=max_batch_size,
        max_image_size=max_image_size,
        seed=seed,
        do_shuffle=do_shuffle,
        use_all=use_all,
        last_method=last_method,
        one_key=one_key,
        return_name=return_name,
        do_test=do_test
    )
    return dataiter