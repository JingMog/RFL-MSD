from __future__ import absolute_import
import os
import sys
import cv2
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(levelname)s] %(name)s -%(message)s',
                    )
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from multiprocessing import Process, synchronize, Lock

from rain import xconfig
from rain.bucket_io import build_data, DataPartioner, build_data_test
from rain.beam_search import BeamSearcher
from rain.initializer import initialize_model
from rain.utils import AccPerformance, collate_torch
from rain.utils import read_image, load_det_sections, parse_title 
import argparse
from tqdm import tqdm
import numpy as np

def filter_list(in_str_list, min_dul_num = 4):

    len_str = len(in_str_list)
    max_dul = len_str // min_dul_num
    is_del = False
    
    
    for n_dul in range(1, max_dul, 1):
        cur_dul_list = in_str_list[-n_dul:]
        cur_n_dul = n_dul + n_dul
        cur_count = 0
        while cur_n_dul <= len_str:
            if in_str_list[-cur_n_dul:(-cur_n_dul + n_dul)] == cur_dul_list:
                cur_count += 1
                cur_n_dul += n_dul
            else:
                break
        
        if cur_count >= min_dul_num:
            is_del = True
            
            in_str_list = in_str_list[:(-cur_n_dul + n_dul)]
            break
    
    
    if is_del:
        return filter_list(in_str_list)
    else:
        return in_str_list

def post_filter(in_file, out_file):

    with open(in_file, 'r') as F:
        in_str = F.readlines()

    out_str = ''
    postfix = 200
    times = 5
    for in_line in in_str:
        cur_line = in_line.strip().split('\t')
        rst_str_list = cur_line[1].split(' ')

        for idx, sstr in enumerate(rst_str_list):
            if sstr == '<s>' or sstr == '</s>':
                rst_str_list = rst_str_list[:idx]
                break

        for _ in range(times):
            rst_str_list = filter_list(rst_str_list)

            c_postfix = min(postfix, len(rst_str_list))
            

            for xt in range(1, c_postfix):
                rst_str_list = filter_list(rst_str_list[:-xt]) + rst_str_list[-xt:]

        cur_line[1] = ' '.join(rst_str_list)
        out_str += '%s\t%s\t%s\n' % (cur_line[0], cur_line[1], cur_line[2])

    with open(out_file, 'w') as F:
        F.write(out_str)

class DataPartionerTest(object):
    def __init__(self, data, size=1, rank=0, seed = 0):
        self.data = data
        self.partitions = []
        self.part_len = len(data) // size
        self.rank = rank
        
        import random
        random.seed(seed)
        random.shuffle(self.data._batchs)
        begin = 0
        for i in range(size - 1):

            self.partitions.append(self.data._batchs[begin:begin + self.part_len])
            begin += self.part_len

        self.partitions.append(self.data._batchs[begin:])


        assert len(self.partitions) == size

    def __len__(self, ):
        return len(self.partitions[self.rank])

    def __getitem__(self, index):
        #index = (index+1) % len(self)
        return self.data.get_item(self.partitions[self.rank], index)


def test(load_epoch, gpu, rank_id, logname, lock=None):

    logger.info("Start testing epoch %d on gpu %d" % (load_epoch, gpu))
    data_set = build_data_test(
                          max_h           = xconfig.test_max_height,
                          max_w           = xconfig.test_max_width, 
                          max_l           = xconfig.test_max_length,
                          fix_batch_size  = xconfig.test_fix_batch_size,
                          max_batch_size  = xconfig.max_batch_size,
                          max_image_size  = xconfig.max_image_size,
                          seed            = xconfig.seed,
                          do_shuffle      = False,
                          use_all         = True,
                          last_method     = 'fill',
                          one_key         = False,
                          return_name     = True,
                          image_list_file = xconfig.test_image_list_path,
                          det_sections_file= xconfig.test_det_sections,
                          normh           = xconfig.test_image_normh,
                          do_test         = True
                          )
    data_partition = DataPartionerTest(data_set, size = test_world_size, rank=rank_id)
    data_loader = torch.utils.data.DataLoader(dataset=data_partition, batch_size=1, num_workers=2,
                                              collate_fn=collate_torch, shuffle = False)

    # prepare
    torch.cuda.set_device(gpu)
    tester_model = BeamSearcher(vocab_size=xconfig.vocab_size, sos=xconfig.sos, eos=xconfig.eos,
                                beam=xconfig.beam, frame_per_char=xconfig.frame_per_char)
    initialize_model(tester_model, xconfig.model_prefix, xconfig.model_type, load_epoch)
    tester_model.cuda()
    tester_model.eval()


    logfile = open(logname + '_nofilter', 'w')
    acc_metric = AccPerformance(ignores={xconfig.sos,xconfig.eos,xconfig.enter})
    names_set = set()
    avg_cost = 0.0
    with torch.no_grad():
        if rank_id == 0:
            data_loader = tqdm(data_loader)
        for data, data_mask, target, target_mask, names_list in data_loader:
            data = data.cuda()
            data_mask = data_mask.cuda()
            batch_size = len(names_list)
            # print(data.shape, data_mask.shape)
            preds_batch, costs_batch = tester_model.search_gpu(data, data_mask)
            
            #import pdb; pdb.set_trace()

            for i in range(batch_size):
                if names_list[i] in names_set:
                    continue
                else:
                    names_set.add(names_list[i])
                #label = target[i, :int(target_mask[i].sum())].tolist()
                pred = preds_batch[i][0]
                cost = costs_batch[i][0]
                #acc_metric.evaluate(label, pred)
                #lab_str = ' '.join([xconfig.vocab.getWord(wid) for wid in label])
                pred = pred[1:-1]
                pre_str = ' '.join([xconfig.vocab.getWord(wid) for wid in pred])
                
                logfile.write('{}\t{}\t{}\n'.format(names_list[i], pre_str, cost))
                avg_cost += cost
                logfile.flush()
    #avg_cost /= len(names_set)
    #logger.info('decode on epoch {}, acc={}, cost={}'.format(load_epoch, acc_metric.get_performance(), avg_cost))
    logfile.close()
    logger.info("End testing epoch %d on gpu %d" % (load_epoch, gpu))

    # post_filter(logname, logname + '_filter')
    post_filter(logname + '_nofilter', logname)



def norm_check(keyfile, det_sections_file=None, key_info_file = None):
    if det_sections_file is not None:
        det_sections = load_det_sections(det_sections_file)
    else:
        det_sections = None
    
    if key_info_file is None:
        key_info_file = keyfile + '_info.txt'
    # if os.path.exists(key_info_file):
    #     return
    fout = open(key_info_file, 'w')
    with open(keyfile, 'r') as fin:
        all_line = fin.readlines()
    for line in tqdm(all_line):
        line_all = line.strip()
        imgfile = line_all
        img_title = parse_title(imgfile)
        if det_sections is not None:
            img_segmentation = det_sections[img_title]
        else:
            img_segmentation = None
        label = '<s> </s>'
        img = read_image(imgfile, xconfig.test_image_normh, img_segmentation, xconfig.img_fix_char_height)
        if img is None:
            logger.info('{} is None'.format(imgfile))
            continue
        vocab = xconfig.vocab
        trans_label = [vocab.getID(l) for l in label.split(' ')]
        item = (imgfile, int(img.shape[1]), int(img.shape[2]), int(len(trans_label)), 0, 0)
        fout.write('{} {} {} {} {} {}\n'.format(imgfile, int(img.shape[1]), int(img.shape[2]),
                                                int(len(trans_label)), 0, 0))
    fout.close()




if __name__ == '__main__':

    parser = argparse.ArgumentParser("OCR MultiProcess MultiCPU Single Epoch Testing")
    parser.add_argument('--process_per_gpu', type=int)
    parser.add_argument('--test_epoch', type=int)
    parser.add_argument('--used_gpu_id', type=str)
    parser.add_argument('--test_image_list_path', type=str)
    parser.add_argument('--img_fix_char_height', type=int, default=None)
    parser.add_argument('--test_det_sections', type=str, default=None)
    
    parser.add_argument('--norm_check_cpus', type=int, default = 16)
    parser.add_argument('--do_norm_check', type=int, default = 1)
    args = parser.parse_args()
    #import pdb; pdb.set_trace()

    xconfig.test_image_list_path = args.test_image_list_path
    xconfig.test_log_out_path = os.path.join( os.path.dirname(args.test_image_list_path), 'pred.trans')

    xconfig.img_fix_char_height = args.img_fix_char_height
    xconfig.test_det_sections = args.test_det_sections

    # norm_check(xconfig.test_image_list_path, args.test_det_sections)
    
    if args.do_norm_check > 0:
        norm_check_name_list = list()
        norm_check_pre_list = list()
        total_check_file = xconfig.test_image_list_path + '_info.txt'

        with open(xconfig.test_image_list_path, 'r') as F:
            all_lines = F.readlines()
        len_images = len(all_lines)
        sep_list_id = list(np.linspace(0, len_images, args.norm_check_cpus + 1, dtype = int))

        for idx in range(args.norm_check_cpus):
            norm_check_name_list.append(total_check_file + '_part%d.txt'%(idx))
            cur_name = xconfig.test_image_list_path + '_part%d.txt'%(idx)
            with open(cur_name, 'w') as F:
                for jdx in range(sep_list_id[idx], sep_list_id[idx + 1], 1):
                    F.write(all_lines[jdx])
            norm_check_pre_list.append(cur_name)

        records = []  
        for f_in, f_out in zip(norm_check_pre_list, norm_check_name_list):

            p = Process(target=norm_check, args=(f_in, args.test_det_sections, f_out))
            p.start()
            records.append(p)

        for p in records:
            p.join()
        cat_str = 'cat'
        for xt in norm_check_name_list:
            cat_str += ' %s'%(xt)
        cat_str += ' > %s'%(total_check_file)
        os.system(cat_str)
        for xt in norm_check_pre_list:
            rm_str = 'rm %s' %(xt)
            os.system(rm_str)
        for xt in norm_check_name_list:
            rm_str = 'rm %s' %(xt)
            os.system(rm_str)

    # import pdb; pdb.set_trace()
    test_epoch = args.test_epoch

    uesd_gpus = [int(g) for g in args.used_gpu_id.split(",")]
    process_per_gpu = args.process_per_gpu

    
    resource_list = []
    log_name_list = []
    part_id = 0
    for igpu in uesd_gpus:
        for icpu in range(process_per_gpu):
            resource_list.append(igpu)
            log_name_list.append('{}_part_{}.log'.format(xconfig.test_log_out_path, part_id))
            part_id += 1

    test_world_size = len(resource_list)


    # start test
    lock  = Lock()
    records = []

    #test(test_epoch, resource_list[0], 0, log_name_list[0], lock)
    
    for i, use_gpu in enumerate(resource_list):

        p = Process(target=test, args=(test_epoch, use_gpu, i, log_name_list[i], lock))
        p.start()
        records.append(p)

    for p in records:
        p.join()

    cat_str = 'cat'
    logname_all = xconfig.test_log_out_path
    for xt in log_name_list:
        cat_str += ' %s'%(xt)
    cat_str += ' > %s'%(logname_all)
    os.system(cat_str)

    for xt in log_name_list:
        rm_str = 'rm %s' %(xt)
        os.system(rm_str)

