# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import numpy
import sys, os
import time
import codecs
import cv2
import codecs
import torch
import json
logger = logging.getLogger()

from PIL import Image, ImageDraw, ImageFont

def counting_char_height(bbox_img, polygon):
    # prepare
    polygon = numpy.asarray(polygon).astype("int32")
    min_rect = cv2.minAreaRect(polygon)
    width, height = min_rect[1]
    if min_rect[2] < -45:
        height, width = min_rect[1]
    if height > width * 4:
        height = width
    bbox_img = bbox_img.copy()
    if len(bbox_img.shape) == 3 and bbox_img.shape[2] == 3:
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2GRAY) # input is RGB from PIL Image

    # get polygon_img
    mask = numpy.zeros(bbox_img.shape[:2], numpy.uint8)
    cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst_img = cv2.bitwise_and(bbox_img, bbox_img, mask=mask)
    bg_img = numpy.ones_like(bbox_img, numpy.uint8)*255
    cv2.bitwise_not(bg_img, bg_img, mask=mask)
    polygon_img = bg_img + dst_img

    # get threshold
    blur_img = cv2.GaussianBlur(bbox_img, (5,5), 0)
    polygon_img = cv2.GaussianBlur(polygon_img, (5,5), 0)
    ret, _ = cv2.threshold(bbox_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, bin_img = cv2.threshold(polygon_img, ret, 255, cv2.THRESH_BINARY)

    # filter contour
    bin_img = 255 - bin_img
    temp_bin_img = bin_img.copy()
    contours, hierarchy = cv2.findContours(temp_bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    temp_bin_img = numpy.zeros_like(temp_bin_img)

    avg_height = 0
    count = 0
    bp_thresh = 0.8 if height < 64 else 0.5
    for i, c in enumerate(contours):
        if hierarchy[0][i,-1] != -1:
            continue
        rect = cv2.boundingRect(c)
        if rect[2] > 5 and rect[3] > 10:
            min_rect = cv2.minAreaRect(c)
            roi_area = bin_img[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
            pixel_num = roi_area.sum() / 255.0
            area = min_rect[1][0] * min_rect[1][1]
            if area == 0:
                continue
            bp_ratio = pixel_num / area
            scale = (rect[2] * rect[3]) / area
            if bp_ratio > bp_thresh or scale > 2:
                continue
            avg_height += rect[3]
            count += 1
    if count == 0:
        return height/3, 1
    avg_height /= count
    return avg_height, count


def norm_by_height(scr_img, norm_h=40, contours=None, img_fix_char_height=None):
    if img_fix_char_height is None:
        if contours is None:
            img_height = scr_img.shape[0]
            img_width = scr_img.shape[1]
            contours = [[[0, 0], [0, img_height], [img_width, img_height], [img_width, 0]]]
        
        total_height = 0
        total_count = 0
        for contour in contours:
            contour = numpy.round(numpy.array(contour)).reshape(-1, 2).astype(numpy.int32)
            cur_image = scr_img.copy()
            cur_mask = numpy.zeros([cur_image.shape[0], cur_image.shape[1]], dtype=numpy.int32)
            cv2.fillPoly(
                cur_mask,
                [contour],
                1
            )
            # cur_image[cur_mask==0] = 255
            x1 = numpy.min(contour[:, 0])
            y1 = numpy.min(contour[:, 1])
            x2 = numpy.max(contour[:, 0])
            y2 = numpy.max(contour[:, 1])
            cur_image = cur_image[y1:y2+1, x1:x2+1]
            contour[:, 0] = contour[:, 0] - x1
            contour[:, 1] = contour[:, 1] - y1
            
            if (cur_image.shape[0] != 0) and (cur_image.shape[1] != 0):
                cur_avg_height, cur_count = counting_char_height(cur_image, contour.tolist())
                total_height += cur_avg_height * cur_count
                total_count += cur_count

        if total_count == 0:
            avg_height = norm_h
        else:
            avg_height = total_height/total_count

        if avg_height == 0:
            avg_height = norm_h
    else:
        avg_height = img_fix_char_height

    ratio = float(norm_h) / avg_height

    shape = scr_img.shape
    if int(shape[1] * ratio) < 5 or int(shape[0] * ratio < 5):
        return None, ratio
    if int(shape[1] * ratio) > 2400 or int(shape[0] * ratio) > 2400:
        ratio_h = 2400 / float(shape[1])
        ratio_w = 2400 / float(shape[0])
        ratio = min(ratio_h, ratio_w)
    
    return cv2.resize(scr_img, (int(shape[1] * ratio), int(shape[0] * ratio))), ratio

def read_image(imgfile, normh, img_segmentation, img_fix_char_height=None):


    img = cv2.imread(imgfile, cv2.IMREAD_COLOR) # bgr

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        return None

    img, ratio = norm_by_height(img,normh,img_segmentation,img_fix_char_height)

    if img is None:
        return None
    img = numpy.array(img)
    img = img.transpose(2, 0, 1)

    return img


class Vocab(object):
    def __init__(self, vocfile, unk_id=0):
        self._word2id = {}
        self._id2word = {}
        self.unk = "\\unk"
        self.unk_id = unk_id
        with open(vocfile) as f:
            index = 0
            for line in f:
                parts = line.split()
                id = 0
                if len(parts) == 2:
                    id = int(parts[1])
                elif len(parts) == 1:
                    id = index
                    index += 1
                else:
                    print('illegal voc line %s' % line)
                    continue
                self._word2id[parts[0]] = id
                self._id2word[id] = parts[0]
        self._vocab_size = max(self._word2id.values()) + 1
        if self._vocab_size != len(self._word2id):
            print('in vocab file {}, vocab_max {} not equal to vocab count {}, maybe empty id or others' \
                .format(vocfile, self._vocab_size, len(self._word2id)))
        
    def getVocSize(self):
        return self._vocab_size

    def getWord(self, id):
        return self._id2word[id] if id in self._id2word else self.unk

    def getID(self, word):
        return self._word2id[word] if word in self._word2id else self.unk_id

    def words2indices(self, words):
        return [self.getID(w) for w in words]
    
    def indices2words(self, id_list):
        return [self.getWord(i) for i in id_list]

    def indices2label(self, id_list):
        words = self.indices2words(id_list)
        return " ".join(words)

    @property
    def word2id(self):
        return self._word2id

    @property
    def id2word(self):
        return self._id2word

    def get_eos(self):
        return self._word2id['</s>']

    def get_sos(self):
        return self._word2id['<s>']




class AccPerformance(object):
    def __init__(self, ignores={}):
        self.reset()
        self._ignores = ignores

    def reset(self):
        self._error = 0
        self._label = 0
        self._num = 0
        self._right = 0
        self._insert = 0
        self._delete = 0
        self._sub = 0

    def _edit_dist(self, label, rec):
        dist_mat = numpy.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
        dist_mat[0, :] = range(len(rec) + 1)
        dist_mat[:, 0] = range(len(label) + 1)
        for i in range(1, len(label) + 1):
            for j in range(1, len(rec) + 1):
                hit_score = dist_mat[i - 1, j - 1] + (label[i - 1] != rec[j - 1])
                ins_score = dist_mat[i, j - 1] + 1
                del_score = dist_mat[i - 1, j] + 1
                dist_mat[i, j] = min(hit_score, ins_score, del_score)

        return len(label), dist_mat[len(label), len(rec)]

    def _edit_dist_ylhu(self, label, rec):
        y = label
        y_hat = rec
        dist = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
        insert = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
        sub = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
        delete = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
        for i in range(len(y) + 1):
            dist[i][0] = i
        for j in range(len(y_hat) + 1):
            dist[0][j] = j

        for i in range(1, len(y) + 1):
            for j in range(1, len(y_hat) + 1):
                if y[i - 1] != y_hat[j - 1]:
                    cost = 1
                else:
                    cost = 0
                deletion_dist = dist[i - 1][j] + 1
                insertion_dist = dist[i][j - 1] + 1
                if cost:
                    substitution_dist = dist[i - 1][j - 1] + 1
                else:
                    substitution_dist = dist[i - 1][j - 1]
                best = min(insertion_dist, deletion_dist,
                           substitution_dist)

                insert_v = insert[i - 1][j - 1]
                sub_v = sub[i - 1][j - 1]
                delete_v = delete[i - 1][j - 1]
                if cost:
                    if insertion_dist < deletion_dist and insertion_dist < substitution_dist:
                        insert_v = insert[i][j - 1] + 1
                        delete_v = delete[i][j - 1]
                        sub_v = sub[i][j - 1]
                    elif deletion_dist < insertion_dist and deletion_dist < substitution_dist:
                        delete_v = delete[i - 1][j] + 1
                        insert_v = insert[i - 1][j]
                        sub_v = sub[i - 1][j]
                    else:
                        sub_v = sub[i - 1][j - 1] + 1

                dist[i][j] = best
                insert[i][j] = insert_v
                sub[i][j] = sub_v
                delete[i][j] = delete_v

        return len(y), dist[-1, -1], insert[-1, -1], delete[-1, -1], sub[-1, -1]

    def evaluate(self, label, rec):
        label = [l for l in label if l not in self._ignores]
        rec = [r for r in rec if r not in self._ignores]

        (l, e, i, d, s) = self._edit_dist_ylhu(label, rec)
        self._error += e
        self._label += l
        self._insert += i
        self._delete += d
        self._sub += s
        self._num += 1
        if e == 0:
            self._right += 1
        return e,l

    def evaluate_ytz(self, label, rec):
        label = [l for l in label if l not in self._ignores]
        rec = [r for r in rec if r not in self._ignores]

        (l, e, i, d, s) = self._edit_dist_ylhu(label, rec)
        self._error += e
        self._label += l
        self._insert += i
        self._delete += d
        self._sub += s
        self._num += 1
        if e == 0:
            self._right += 1
        return e,l,i,d,s

    def get_performance(self):
        if self._label < 1:
            logger.warn('bad acc performance with label_num={}'.format(self._label))
            self._label = 1
        wer = float(self._error) / self._label
        sacc = float(self._right) / self._num
        logger.info('label={}, error={}, wer={}, SACC={}'.
                    format(self._label, self._error, wer, sacc))
        logger.info('Terror={}, insert={}, delete={}, sub={}'.
                    format(self._error, self._insert, self._delete, self._sub))
        acc = 1 - wer
        return acc, sacc


def parse_title(path):
    name = os.path.basename(path)
    if '.' in name:
        title = name[:name.rindex('.')]
    else:
        title = name
    return title


def load_det_sections(det_sections_file):
    import copy
    import numpy as np
    det_sections = dict()
    for image_name, image_sections in json.load(open(det_sections_file)).items():
        for section_idx, section in enumerate(image_sections):
            section_name = '%s_%d' % (parse_title(image_name), section_idx)
            if section['segmentation'] is None:
                section_name = parse_title(image_name)
                x1 = 0
                y1 = 0
            else:
                x1, y1 = section['bbox'][:2]
            if 'lines' not in section:
                section['lines'] = copy.deepcopy(section['segmentation'])
            segmentation = list()
            for contour in section['lines']:
                contour = np.array(contour).reshape(-1, 2)
                contour[:, 0] = contour[:, 0] - x1
                contour[:, 1] = contour[:, 1] - y1
                segmentation.append(contour.tolist())
            det_sections[section_name] = segmentation
    return det_sections

def collate_torch(batch):
    assert(len(batch) == 1)
    data,data_mask,target,target_mask = batch[0][:4]
    data        = torch.from_numpy(data)
    data_mask   = torch.from_numpy(data_mask)
    target      = torch.from_numpy(target)
    target_mask = torch.from_numpy(target_mask)
    outputs = [data,data_mask,target,target_mask] + batch[0][4:]
    return outputs

def collate_torch_dict(batch):
    assert(len(batch) == 1)
    out_dict = batch[0]
    for key, value in out_dict.items():
        if isinstance(value, numpy.ndarray):
            out_dict[key] = torch.from_numpy(value)
    # data        = torch.from_numpy(data)
    # data_mask   = torch.from_numpy(data_mask)
    # target      = torch.from_numpy(target)
    # target_mask = torch.from_numpy(target_mask)
    # outputs = [data,data_mask,target,target_mask] + batch[0][4:]
    return out_dict
