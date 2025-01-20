from __future__ import absolute_import
import os
import sys
import cv2
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s[%(levelname)s] %(name)s -%(message)s',
                    )
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from multiprocessing import Process, synchronize, Lock
from rain import xconfig
from rain.bucket_io import build_data, DataPartioner, build_data_test
from rain.beam_search import BeamSearcher
from rain.initializer import initialize_model, initialize_model_from_pytorch_v2
from rain.utils import AccPerformance, collate_torch_dict
from rain.utils import read_image, load_det_sections, parse_title
import argparse
from tqdm import tqdm
import Levenshtein
import numpy as np
from rain.calculate_dis import cal_edit_ops, count_ops
import rain.utils as utils
import pdb
import cv2

cv2.setNumThreads(0)

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

    with open(out_file[:-4] + 'post' +out_file[-4:], 'w') as F:
        F.write(out_str)


def line_post_filter(rst_str_list):
    postfix = 200
    times = 5
    for _ in range(times):
        rst_str_list = filter_list(rst_str_list)
        c_postfix = min(postfix, len(rst_str_list))
            
        for xt in range(1, c_postfix):
            rst_str_list = filter_list(rst_str_list[:-xt]) + rst_str_list[-xt:]
    return rst_str_list

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


def cal_edit_ops(str1, str2):
    char_idx_dict = dict()
    for item in str1:
        if item not in char_idx_dict:
            char_idx_dict[item] = chr(len(char_idx_dict)) 
    for item in str2:
        if item not in char_idx_dict:
            char_idx_dict[item] = chr(len(char_idx_dict))
    str1 = ''.join([char_idx_dict[item] for item in str1])
    str2 = ''.join([char_idx_dict[item] for item in str2])
    ops = Levenshtein.editops(str1, str2) 
    return ops


def count_ops(ops):
    insert_nums = sum([1 for op_name, *_ in ops if op_name=='delete'])
    substitute_nums = sum([1 for op_name, *_ in ops if op_name=='replace'])
    delete_nums = sum([1 for op_name, *_ in ops if op_name=='insert'])
    assert delete_nums + substitute_nums + insert_nums == len(ops)
    return delete_nums, substitute_nums, insert_nums

def _edit_dist(label, rec):
    dist_mat = np.zeros((len(label)+1, len(rec) +1), dtype='int32')
    dist_mat[0,:]= range(len(rec) +1)
    dist_mat[:,0] = range(len(label) +1)
    for i in range(1, len(label)+1):
        for j in range(1, len(rec) +1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
                
    return len(label), dist_mat[len(label), len(rec)]

def calc_edit_dist(label, rec):
    label =[l for l in label]
    rec = [r for r in rec]
    return _edit_dist(label, rec)[1]

def rm_bracket(label_list, rm_list = ['\\underline']):
    if(len(label_list) == 0):
        return 0, []
    if(label_list[0] != '{'):
        return 0, []
    left_bracket_num = 0
    right_bracket_num = 0
    for i, temp in enumerate(label_list):
        if(temp == '{'):
            left_bracket_num += 1
        elif (temp == '}'):
            right_bracket_num += 1
        if(left_bracket_num == right_bracket_num):
            temp_list = rm_underline_textbf(label_list[1:i], rm_list)
            return i + 1, temp_list
    return 0, []


def rm_underline_textbf(label_list, rm_list = ['\\underline']):
    # print(label_list)
    new_label_list = []
    idx = 0
    while idx < len(label_list):
        if label_list[idx] != rm_list[0]:
            new_label_list.append(label_list[idx])
            idx += 1
        else:
            new_idx, temp_list = rm_bracket(label_list[idx+1:])
            new_label_list = new_label_list + temp_list
            idx += 1
            idx += new_idx
    return new_label_list

def test(load_epoch, gpu, rank_id, logname, lock=None, label_path=None, h=10, w=80, args=None):
    logger.info("Start testing epoch %d on gpu %d" % (load_epoch, gpu))
    data_set = build_data(lrcfile        = xconfig.test_lrc,
                          keyfile          = xconfig.test_lrc_cache,
                          max_h            = xconfig.test_max_height,
                          max_w            = xconfig.test_max_width, 
                          max_l            = xconfig.test_max_length,
                          fix_batch_size   = xconfig.test_fix_batch_size,
                          max_batch_size   = xconfig.max_batch_size,
                          max_image_size   = xconfig.max_image_size,
                          seed             = xconfig.seed,
                          do_shuffle       = False,
                          use_all          = True,
                          last_method      = 'fill',
                          one_key          = False,
                          return_name      = True,
                          image_list_file  = None,
                          #det_sections_file= xconfig.test_det_sections,
                          normh            = xconfig.test_lrc_normh,
                          do_test          = True,
                          #label_path       = label_path
                          )
    data_partition = DataPartionerTest(data_set, size = test_world_size, rank=rank_id)
    data_loader = torch.utils.data.DataLoader(dataset=data_partition, batch_size=1, num_workers=8,
                                              collate_fn=collate_torch_dict, shuffle = False)
    
    # prepare
    torch.cuda.set_device(gpu)
    tester_model = BeamSearcher(vocab_size=xconfig.vocab_size, sos=xconfig.sos, eos=xconfig.eos,
                                beam=xconfig.beam, frame_per_char=xconfig.frame_per_char)
    if args.load_param_path is None or os.path.exists(args.load_param_path) is False:
        initialize_model(tester_model, xconfig.model_prefix, xconfig.model_type, load_epoch, allow_missing=False)
    else:
        initialize_model_from_pytorch_v2(tester_model, args.load_param_path)
    tester_model.cuda()
    tester_model.eval()
    
    acc_metric = AccPerformance(ignores={xconfig.sos,xconfig.eos,xconfig.enter})
    names_set = set()
    avg_cost = 0.0
    n = 0
    cum_n = 0
    top1n = 0
    count = 0
    corr_count = 0
    top1corr_count = 0
    
    d = 0
    s = 0
    insert = 0
    
    d_top3 = 0
    s_top3 = 0
    i_top3 = 0


    avg_h = []
    avg_w = []
    top1_lines = []
    top3_lines = []

    prefix, ext = os.path.splitext(logname)
    logname_cs_top1 = "{}_cs_top1.txt".format(prefix)
    logname_top1 = "{}_top1.txt".format(prefix)
    logname_top3 = "{}_top3.txt".format(prefix)
    logname_top3_all = "{}_top3_all.txt".format(prefix)
    fout_cs_top1 = open(logname_cs_top1, "w")
    fout_top1 = open(logname_top1, "w")
    fout_top3 = open(logname_top3, "w")
    fout_top3_all = open(logname_top3_all, "w")
    with torch.no_grad():
        if rank_id == 0:
            data_loader = tqdm(data_loader)
        
        count_sample = 0
        
        for bid, data_dict in enumerate(data_loader):
            # if data_dict["names_list"][0] not in ['2499-0-0-0', '48-0-0-0', '2001-0-0-0']:
            #     continue
            # ori_label = xconfig.vocab.indices2words(data_dict['target'][0].tolist())
            # ori_label = " ".join(ori_label)
            for key in data_dict:
                value = data_dict[key]
                if not isinstance(value, torch.Tensor):
                    continue
                if key.find("mask") != -1:
                    data_dict[key] = value.cuda().detach()
                else:
                    data_dict[key] = value.cuda()
            
            data = data_dict["data"]
            data_mask = data_dict["data_mask"]
            names_list = data_dict["names_list"]
            b, c, h, w = data.shape
            avg_h.append(h)
            avg_w.append(w)
            
            batch_size = len(names_list)
            
            if len(names_list) == 0:
                continue
            
            img_info = data_set._data_parser.record_loader.get_record(int(names_list[0].replace('-0-0-0', '')))
            img_path = img_info['src_image_path']
            img_name = [img_path[-15:]]

            if args.viz_path:
                if args.cur_viz_num > args.viz_num:
                    break
            try:
                rets = tester_model.search_gpu(data, data_mask, args=args, names_list=img_name, is_show=args.is_show)
                preds_batch = rets[0]
                costs_batch = rets[1]
                cs_string_batch = rets[2]

            except BaseException as e:
                preds_batch = [ [ ["<s>","</s>"], ["<s>","</s>"], ["<s>","</s>"] ] for _ in range(batch_size)]
                costs_batch = [ [ 1e10, 1e10, 1e10 ] for _ in range(batch_size)]
                print("catch exception for name_lists = {}".format(names_list))
                print("err_info = {}".format(e))
            
            for i in range(batch_size):
                if names_list[i] in names_set:
                    continue
                else:
                    names_set.add(names_list[i])
                count += 1
                # target_str = [int(temp) for temp in target[i][:int(target_mask[i].sum())].tolist()[1:-1] if temp != xconfig.vocab.getID("\\smear") and temp != xconfig.vocab.getID("\\space")]
                # pred_str = [int(temp) for temp in preds_batch[i][0][1:-1] if temp != xconfig.vocab.getID("\\smear") and temp != xconfig.vocab.getID("\\space")]
                # pred_str = rm_underline_textbf(pred_str)
                target_str = ["\\smear"]
                pred_str = [temp for temp in preds_batch[i][0][:-1] if temp != "\\smear" and temp != "\\space"]
                if(len(pred_str) > 0):
                    pred_str = rm_underline_textbf(pred_str)
                if(len(target_str) > 0):
                    target_str = rm_underline_textbf(target_str)
                t = calc_edit_dist(label=target_str, rec=pred_str)
                # pdb.set_trace()
                fout_cs_top1.write("{}\t{}\n".format(names_list[i], " ".join(cs_string_batch[i][0])))
                fout_cs_top1.flush()
                fout_top1.write("{}\t{}\t{}\n".format(names_list[i], " ".join(target_str), " ".join(pred_str)))
                fout_top1.flush()
                """if('\\in' in ' '.join(target_str_temp)):
                    print(' '.join(target_str_temp))
                    print(' '.join(pred_str_temp))"""
                cur_ops = cal_edit_ops(pred_str, target_str)
                cur_d, cur_s, cur_i = count_ops(cur_ops)
                d += cur_d
                s += cur_s
                insert += cur_i

                top1n += t
                if(t == 0):
                    top1corr_count += 1


                min_dist = np.inf
                min_idx = -1
                num = 0
                all_preds = []
                for j, (pred, cost) in enumerate(zip(preds_batch[i], costs_batch[i])):
                    # if j > 2:
                    #     continue
                    num += 1
                    pred_str = [temp for temp in pred[:-1] if temp != "\\smear" and temp != "\\space"]
                    if(len(pred_str) > 0):
                        pred_str = rm_underline_textbf(pred_str)
                    all_preds.append([x for x in pred_str])
                    # pred_str = [int(temp) for temp in pred[:-1] if temp != "\\smear" and temp != "\\space"]
                    # pred_str = rm_underline_textbf(pred_str)
                    t = calc_edit_dist(label=target_str, rec=pred_str)
                    if (t < min_dist):
                        min_dist = t
                        min_idx = j
                    pred = pred[:-1]
                if min_dist == np.inf:
                    min_dist = np.inf
                    min_idx = -1
                # print(num)
                pred_str = [temp for temp in preds_batch[i][min_idx][:-1] if temp != "\\smear" and temp != "\\space"]
                # pred_str = [int(temp) for temp in preds_batch[i][min_idx][:-1] if temp != "\\smear" and temp != "\\space"]
                # pred_str = rm_underline_textbf(pred_str)
                if(len(pred_str) > 0):
                    pred_str = rm_underline_textbf(pred_str)
                fout_top3.write("{}\t{}\t{}\n".format(names_list[i], " ".join(target_str), " ".join(pred_str)))
                fout_top3.flush()
                fout_top3_all.write("{}\t{}".format(names_list[i], " ".join(target_str)))
                for cur_pred_arr in all_preds:
                    fout_top3_all.write("\t{}".format(" ".join(cur_pred_arr)))
                fout_top3_all.write("\n")
                fout_top3_all.flush()

                cur_ops = cal_edit_ops(pred_str, target_str)
                cur_d, cur_s, cur_i = count_ops(cur_ops)
                d_top3 += cur_d
                s_top3 += cur_s
                i_top3 += cur_i
                n += min_dist
                cum_n += len(target_str)
                if(min_dist == 0):
                    corr_count += 1

    fout_cs_top1.close()
    fout_top1.close()
    fout_top3.close()
    fout_top3_all.close()
    with open(logname, 'w') as fp:
        fp.write(str(top1n) + '\t' + str(n) + '\t' + str(cum_n) + '\t' + str(top1corr_count) + '\t' + str(corr_count) + '\t' + str(count) + '\n')
        fp.write(str(d) + '\t' + str(s) + '\t' + str(insert) + '\t' + str(d_top3) + '\t' + str(s_top3) + '\t' + str(i_top3) + '\n')

    logger.info("End testing epoch %d on gpu %d" % (load_epoch, gpu))

def get_label_dict(label_path=None):
    if label_path is not None:
        label_dict = dict()
        label_lines = open(label_path).readlines()
        for line in label_lines:
            if(len(line.split('\t')) == 3):
                path, json, seg = line.split('\t')
            else:
                path, seg = line.split('\t')
            seg = seg.strip()
            img_title = parse_title(path)
            # print(img_title)
            if(img_title not in label_dict):
                label_dict.update({img_title: seg})
        return label_dict
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OCR MultiProcess MultiCPU Single Epoch Testing")
    parser.add_argument('--process_per_gpu', type=int, default=1)
    parser.add_argument('--used_gpu_id', type=str, default='2')
    parser.add_argument('--test_image_list_path', type=str, default=None)
    parser.add_argument('--test_lrc', type=str)
    parser.add_argument('--test_lrc_cache', type=str, default=None)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--h', type=int, default=10)
    parser.add_argument('--w', type=int, default=80)
    parser.add_argument('--test_det_sections', type=str, default=None)
    parser.add_argument('--load_param_path', type=str, default=None)
    parser.add_argument('--load_epoch', type=int, default=None)
    parser.add_argument('--is_show', type=str, default="False")
    parser.add_argument('--viz_path', type=str, default=None)
    parser.add_argument('--viz_num', type=int, default=10)
    args = parser.parse_args()
    
    args.cur_viz_num = 0
    if args.is_show == 'True':
        args.is_show = True
    else:
        args.is_show = False

    if args.test_lrc is not None:
        xconfig.test_lrc = args.test_lrc
        if args.test_lrc_cache is None:
            xconfig.test_lrc_cache = xconfig.test_lrc + ".line.cache"
        else:
            xconfig.test_lrc_cache = args.test_lrc_cache

    if args.name in ["math_all_wo_rotate", "math_划题识别联调", "math_online", "math_photo"]:
        xconfig.dst_type_dict["TopicType"] = ["FillBlank"]
    #xconfig.test_log_out_path = os.path.join(os.path.dirname(args.test_image_list_path), 'pred.trans')
    test_epoch = 0
    if args.load_epoch is None:
        test_epochs = []
        for les in xconfig.test_load_epochs.split(","):
            if "-" in les:
                les1_list = les.split("-")
                assert len(les1_list) == 2, "test_load_epochs %s format error" % xconfig.test_load_epochs
                test_epochs += list(range(int(les1_list[0]), int(les1_list[1])))
            else:
                test_epochs.append(int(les))
        test_epoch = test_epochs[0]
    else:
        test_epoch = args.load_epoch
    
    xconfig.test_log_out_path = os.path.join(xconfig.base_model_dir, "test[%s]_epoch[%d]"%(args.name, test_epoch))
    if not os.path.exists(xconfig.test_log_out_path):
        os.makedirs(xconfig.test_log_out_path)
    xconfig.test_log_out_path = os.path.join(xconfig.test_log_out_path, "test[%s]_epoch[%d].trans"%(args.name, test_epoch))
    print(xconfig.test_log_out_path)
        
    uesd_gpus = [int(g) for g in args.used_gpu_id.split("_")]
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

    # test(test_epoch, resource_list[0], 0, log_name_list[0], lock)
    
    for i, use_gpu in enumerate(resource_list):
        if i == 0:
            continue
        p = Process(target=test, args=(test_epoch, use_gpu, i, log_name_list[i], lock, args.label_path, args.h, args.w, args))
        p.start()
        records.append(p)
    i = 0
    use_gpu = resource_list[i]
    test(test_epoch, use_gpu, i, log_name_list[i], lock, args.label_path, args.h, args.w, args)

    for p in records:
        p.join()

    cat_str = 'cat'
    logname_all = xconfig.test_log_out_path
    for xt in log_name_list:
        cat_str += ' %s'%(xt)
    cat_str += ' > %s'%(logname_all)
    os.system(cat_str)

    cat_str = 'cat'
    logname_all = xconfig.test_log_out_path
    prefix, ext = os.path.splitext(logname_all)
    logname_all_top1 = "{}_top1.txt".format(prefix)
    for xt in log_name_list:
        prefix2, ext2 = os.path.splitext(xt)
        logname_top1 = "{}_top1.txt".format(prefix2)
        cat_str += ' %s'%(logname_top1)
    cat_str += ' > %s'%(logname_all_top1)
    os.system(cat_str)

    cat_str = 'cat'
    logname_all = xconfig.test_log_out_path
    prefix, ext = os.path.splitext(logname_all)
    cs_all_top1 = "{}_cs_top1.txt".format(prefix)
    for xt in log_name_list:
        prefix2, ext2 = os.path.splitext(xt)
        cs_top1 = "{}_cs_top1.txt".format(prefix2)
        cat_str += ' %s'%(cs_top1)
    cat_str += ' > %s'%(cs_all_top1)
    os.system(cat_str)

    cat_str = 'cat'
    logname_all = xconfig.test_log_out_path
    prefix, ext = os.path.splitext(logname_all)
    logname_all_top3 = "{}_top3.txt".format(prefix)
    for xt in log_name_list:
        prefix2, ext2 = os.path.splitext(xt)
        logname_top3 = "{}_top3.txt".format(prefix2)
        cat_str += ' %s'%(logname_top3)
    cat_str += ' > %s'%(logname_all_top3)
    os.system(cat_str)

    cat_str = 'cat'
    logname_all = xconfig.test_log_out_path
    prefix, ext = os.path.splitext(logname_all)
    logname_all_top3_all = "{}_top3_all.txt".format(prefix)
    for xt in log_name_list:
        prefix2, ext2 = os.path.splitext(xt)
        logname_top3_all = "{}_top3_all.txt".format(prefix2)
        cat_str += ' %s'%(logname_top3_all)
    cat_str += ' > %s'%(logname_all_top3_all)
    os.system(cat_str)

    post_tool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_process_chemfig.py")

    os.system("python refine_name_for_log.py -lrc_path {} -addPath 1 -key src_image_path -input {} -output {}".format(xconfig.test_lrc, logname_all_top1, logname_all_top1.replace(".txt", ".wName.txt")))
    os.system("python {} -input {}".format(post_tool_path, logname_all_top1.replace(".txt", ".wName.txt")))

    os.system("python refine_name_for_log.py -lrc_path {} -addPath 1 -key src_image_path -input {} -output {}".format(xconfig.test_lrc, logname_all_top3, logname_all_top3.replace(".txt", ".wName.txt")))
    os.system("python {} -input {}".format(post_tool_path, logname_all_top3.replace(".txt", ".wName.txt")))

    os.system("python refine_name_for_log.py -lrc_path {} -addPath 1 -key src_image_path -input {} -output {}".format(xconfig.test_lrc, cs_all_top1, cs_all_top1.replace(".txt", ".wName.txt")))
    os.system("python {} -input {}".format(post_tool_path, cs_all_top1.replace(".txt", ".wName.txt")))

    lines = open(logname_all).readlines()
    top1n = 0
    top3n = 0
    n = 0
    top1count = 0
    top3count = 0
    count = 0

    d = 0
    s = 0
    insert = 0
    d_top3 = 0
    s_top3 = 0
    i_top3 = 0
    for i, line in enumerate(lines):
        if i % 2 == 0:
            line = line.strip().split('\t')
            top1n += int(line[0])
            top3n += int(line[1])
            n += int(line[2])
            top1count += int(line[3])
            top3count += int(line[4])
            count += int(line[5])
        else:
            line = line.strip().split('\t')
            d += int(line[0])
            s += int(line[1])
            insert += int(line[2])
            d_top3 += int(line[3])
            s_top3 += int(line[4])
            i_top3 += int(line[5])
    if n==0:
        n+=1e-10
    print('top1 wacc:%f\tD = %d\tS = %d\tI= %d'%(1 - top1n / n, d, s, insert))
    print('top1 sacc:%f' % (top1count / count))
    print('top3 wacc:%f\tD = %d\tS = %d\tI= %d'%(1 - top3n / n, d_top3, s_top3, i_top3))
    print('top3 sacc:%f' % (top3count / count))

    for xt in log_name_list:
        rm_str = 'rm %s' %(xt)
        os.system(rm_str)
        prefix2, ext2 = os.path.splitext(xt)
        logname_cs_top = "{}_cs_top1.txt".format(prefix2)
        logname_top1 = "{}_top1.txt".format(prefix2)
        logname_top3 = "{}_top3.txt".format(prefix2)
        logname_top3_all = "{}_top3_all.txt".format(prefix2)
        os.system('rm %s' % (logname_cs_top))
        os.system('rm %s' % (logname_top1))
        os.system('rm %s' % (logname_top3))
        os.system('rm %s' % (logname_top3_all))
