import os, sys
import pdb
import argparse
import Levenshtein
import chemfig_struct
import chemfig_ops
import chemfig_parser
import graph_cmp
import tqdm
import shutil
import cv2
from multiprocessing import Process, synchronize, Lock, Manager, Pool
import multiprocessing
from six.moves import queue
import warnings
from viz_struct import *
from Tool_Formula.latex_norm.transcription import parse_transcription
from collections import OrderedDict
import utils
from chemfig_struct import *

def chemfig_parse_eval(inputStr, echo=False):
    inputStr = chemfig_parser.process_escape_strs(inputStr, is_process_virtual_bond=False)
    units = chemfig_parser.parse_units(inputStr)
    elements = chemfig_parser.parse_element(units)
    root_atom, _ = chemfig_parser.connect_relation(elements, echo=echo)
    # try:
    all_atoms = text_render.pre_process(root_atom)
    # except:
    #     print(inputStr)
    
    for atom in all_atoms:
        try:
            text_arr = parse_transcription(atom.m_text.replace("\\enter", "").replace("\\space", " ").replace("\\\\", ""), simple_trans=True)
        except:
            print("error atom: ", atom)
            text_arr = parse_transcription(atom.m_text.replace("\\enter", "").replace("\\space", " ").replace("\\\\", ""), simple_trans=True)
            

        text_arr = utils.post_process(text_arr)
        atom.m_text = "".join(text_arr)
        for bond in atom.in_bonds + atom.out_bonds:
            if bond.m_extra_info is not None and bond.m_extra_info.replace(" ","") == "draw=none":
                bond.begin_atom.out_bonds.remove(bond)
                bond.end_atom.in_bonds.remove(bond)
        
    
    new_all_atoms = []
    for atom in all_atoms:
        if len(atom.in_bonds+atom.out_bonds) == 0 and len(all_atoms) > 1:
            continue
        new_all_atoms.append(atom)
    root_atom = new_all_atoms[0]
    return root_atom


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
    insert_nums = sum([1 for op_name, *_ in ops if op_name == 'delete'])
    substitute_nums = sum([1 for op_name, *_ in ops if op_name == 'replace'])
    delete_nums = sum([1 for op_name, *_ in ops if op_name == 'insert'])
    assert delete_nums + substitute_nums + insert_nums == len(ops)
    return delete_nums, substitute_nums, insert_nums


def norm_text(text_arr, do_norm_sub=True):
    tmp_text_arr = []
    for text in text_arr:
        if text.startswith("[:") and text.endswith("]"):
            continue
        if text.find("[:") != -1 and text.endswith("]"):
            pos = text.find("[:")
            text = text[:pos]
        if text == "\\:" or text == ":":
            text = ":"
        tmp_text_arr.append(text)
    if do_norm_sub > 0:
        out_text_arr = utils.norm_sub(tmp_text_arr)
    else:
        out_text_arr = tmp_text_arr
    return out_text_arr


def compare_struct_v3(lab_rep_dict, rec_rep_dict):
    ret = 0
    extra_cnt = 0
    lab_keys = set(list(lab_rep_dict.keys()))
    lab_cnt = len(lab_keys)

    all_keys = []
    all_keys += list(lab_rep_dict.keys())
    all_keys += list(rec_rep_dict.keys())
    all_keys = list(set(all_keys))

    for key in all_keys:
        if key in rec_rep_dict and key not in lab_rep_dict:
            extra_cnt += 1
        if key not in rec_rep_dict:
            rec_rep_dict[key] = {"text": "\\smear", "root": None}
        if key not in lab_rep_dict:
            lab_rep_dict[key] = {"text": "\\smear", "root": None}

    correct_cnt = 0
    for key in lab_rep_dict:
        if key not in rec_rep_dict:
            ret = 1
            continue
        lab_text = lab_rep_dict[key]["text"]
        lab_atom = lab_rep_dict[key]["root"]
        rec_text = rec_rep_dict[key]["text"]
        rec_atom = rec_rep_dict[key]["root"]
        
        if lab_atom and rec_atom:
            res_graph = graph_cmp.compare_graph(lab_atom, rec_atom)
            rec_rep_dict[key]["res_graph"] = res_graph
            rec_rep_dict[key]["res_cmp"] = res_graph
            rec_rep_dict[key]["lab_atom"] = lab_atom
            rec_rep_dict[key]["rec_atom"] = rec_atom
            rec_rep_dict[key]["lab_text"] = lab_text
            rec_rep_dict[key]["rec_text"] = rec_text
            if res_graph == 1:
                ret = 1
            else:
                if key in lab_keys:
                    correct_cnt += 1
        else:
            res_text = 0 if lab_text.replace(" ", "") == rec_text.replace(" ", "") else 1
            rec_rep_dict[key]["res_text"] = res_text
            rec_rep_dict[key]["res_cmp"] = res_text
            rec_rep_dict[key]["lab_text"] = lab_text
            rec_rep_dict[key]["rec_text"] = rec_text
            if res_text == 1:
                ret = 1
            else:
                if key in lab_keys:
                    correct_cnt += 1
    return ret, lab_cnt, correct_cnt, extra_cnt

def parse_lab_rep_dict(lab_rep_dict: OrderedDict, is_process_virtual_bond=True):
    for key, value in lab_rep_dict.items():
        begin = value.find("{")
        end = value.rfind("}")
        if begin == -1 or end == -1:
            root_atom = None
            out_text = value
            lab_rep_dict[key] = {"text": "\\chemfig { " + out_text + " }", "root": None}
            raise ValueError("can not find bracket pair in chemfig domain, {}".format(vaue))
        elif value.find("\\Charge") != -1:
            in_value = value[begin + 1:end].replace("\\\\", "").replace("\\enter", "").replace("\\space", " ")
            root_atom = chemfig_parse_eval(in_value)
            out_text_arr = text_render.rend_text(root_atom)
            out_text = " ".join(out_text_arr)
            lab_rep_dict[key] = {"text": "\\chemfig { " + out_text + " }", "root": None}
            #continue
        else:
            in_value = value[begin + 1:end].replace("\\\\", "").replace("\\enter", "").replace("\\space", " ")
            if is_process_virtual_bond:
                in_value = chemfig_parser.process_virtual_bond(in_value)
            root_atom = chemfig_parse_eval(in_value)
            out_text = " ".join(text_render.rend_text(root_atom))
            lab_rep_dict[key] = {"text": "\\chemfig { " + out_text + " }", "root": root_atom}

def remove_fake_chemfig(rep_dict: OrderedDict, global_text=""):
    new_items = []
    new_global_text = global_text
    for key, value in rep_dict.items():
        text = value["text"]
        atom = value["root"]
        if atom is not None:
            #ret = chemfig_render.norm_flat_chemfig(atom)
            ret = chemfig_ops.NormFlatChemfig(atom)
        else:
            ret = None
        if isinstance(ret, Atom) or ret is None:
            new_items.append((key, value))
        else:
            new_text = " ".join(ret)
            new_global_text = new_global_text.replace(key, new_text)
    ind = 0
    key_map = {}
    new_rep_dict = OrderedDict()

    new_text_chemfig = []
    for key, value in new_items:
        new_key = "\\chem{}".format(chr(ord("a") + ind))
        ind += 1
        key_map[key] = new_key
        new_rep_dict[new_key] = value
        new_text_chemfig.append(value["text"])
    # pdb.set_trace()
    new_text_chemfig = " ".join(new_text_chemfig) 

    text_arr = new_global_text.split(" ")
    for i, text in enumerate(text_arr):
        if text in key_map:
            text_arr[i] = key_map[text]
    new_global_text = " ".join(text_arr)

    # new_text_chemfig = " ".join( list(new_rep_dict.values()) )
    return new_text_chemfig, new_rep_dict, new_global_text

def process_chemfig_str(inputStr, is_origin = False):
    chem_trans, rep_dict, rep_trans = utils.replace_chemfig(inputStr)
    words = parse_transcription(rep_trans, simple_trans=True)
    words = utils.post_process(words)
    new_words = []
    for x in words:
        if x in rep_dict:
            chem_str = rep_dict[x]
            if chem_str.replace(" ", "") != "\\chemfig{}":
                new_words.append(chem_str)
        else:
            new_words.append(x)
    chem_trans, rep_dict, rep_trans = utils.replace_chemfig(" ".join(new_words))
    parse_lab_rep_dict(rep_dict, False)
    chem_trans, rep_dict, rep_trans = remove_fake_chemfig(rep_dict, rep_trans)
    return chem_trans, rep_dict, rep_trans

def do_single_task(line, line_id, args, ref_set, \
        m_metrics, m_metrics_lock, \
        records_queue, records_queue_lock, \
        m_err_lines = None, m_err_lines_lock = None, \
        m_exception_lines = None, m_exception_lines_lock = None, \
        m_post_lines = None, m_post_lines_lock = None
    ):
    # reset id
    chemfig_struct.Atom.index = 0
    chemfig_struct.Bond.index = 0

    spts = line.strip("\n").split("\t")
    if len(spts) < 3:
        print("err line = %s" % line.strip())
        return

    data_key = spts[0]
    lab = spts[1]
    recs = spts[2:]

    img_path = data_key.replace("-0-0-0", "")
    data_key = os.path.splitext(os.path.basename(img_path))[0]

    lab_arr = lab.split(" ")
    lab_arr = list(filter(None, lab_arr))
    rec_arrs = [list(filter(None, rec.split(" "))) for rec in recs]

    all_ops = [(rec_arr, cal_edit_ops(rec_arr, lab_arr)) for rec_arr in rec_arrs]
    all_ops = sorted(all_ops, key=lambda x: len(x[1]))
    base_ops = all_ops[0][1]

    # struct
    rec_chemfigs = []
    for rec in recs:
        try:
            _rec_chemfig, _rec_rep_dict, _rec_global_text = process_chemfig_str(rec)
        except:
            # print("error rec!!!")
            _rec_chemfig = ""
            _rec_rep_dict = {}
            _rec_global_text = "\\smear"
        rec_chemfigs.append((_rec_chemfig, _rec_rep_dict, _rec_global_text))
		
    if args.origin_trans_dict is not None and args.origin_cmp > 0:
        if data_key in args.origin_trans_dict:
            ori_trans = args.origin_trans_dict[data_key]
            lab_chemfig, lab_rep_dict, lab_global_text = process_chemfig_str(ori_trans)
        else:
            with records_queue_lock:
                records_queue.put(1)
            return
    else:
        lab_chemfig, lab_rep_dict, lab_global_text  = process_chemfig_str(lab)


    all_ops = [(rec_rep_dict, compare_struct_v3(lab_rep_dict, rec_rep_dict), rec_global_text) for _, rec_rep_dict, rec_global_text in rec_chemfigs]
    all_ops = sorted(all_ops, key=lambda x: x[1][1] - x[1][2])  #ret, lab_cnt, correct_cnt, extra_cnt
    cmp_res, single_n, single_acc, single_extra = all_ops[0][1]
    cmp_rec_global_text = all_ops[0][2]
    cmp_lab_global_text = lab_global_text

    # save post log
    if m_post_lines_lock:
        tgt_lab_global_text = lab_global_text
        tgt_rec_global_text = all_ops[0][2]
        with m_post_lines_lock:
            m_post_lines.put("{}\t{}\t{}\n".format(spts[0], tgt_lab_global_text, tgt_rec_global_text))

    # do viz struct
    tgt_rec_rep_dict = all_ops[0][0]
    if args.viz > 0:
        img_key = data_key
        if img_key in args.origin_trans_dict:
            ori_trans = args.origin_trans_dict[img_key]
        else:
            ori_trans = None
        if len(tgt_rec_rep_dict) == 0:
            tgt_rec_rep_dict = {"\\chema": {"res_text": cmp_res, "res_cmp": cmp_res, "lab_text": "\\smear", "rec_text": "\\smear"}}
        
        viz_img = viz_struct_res(img_path, tgt_rec_rep_dict, ori_trans)
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if args.refLog is not None and os.path.exists(args.refLog) and img_key in ref_set:
            output_path = os.path.join(args.output, "RefSet/res[{}]/{}.jpg".format(cmp_res, img_name))
        else:
            output_path = os.path.join(args.output, "res[{}]/{}.jpg".format(cmp_res, img_name))
        outDir = os.path.dirname(output_path)
        if not os.path.exists(outDir):
            os.makedirs(outDir, exist_ok=True)
        cv2.imwrite(output_path, viz_img)
        # pdb.set_trace()
        # shutil.copy(output_path, "debug.jpg")
        # print(lab)
        # print(recs)
        # print("----------{}------------".format(cmp_res))
        # pdb.set_trace()

    ref_cmp_res = None
    with m_metrics_lock:
        #update base
        cur_metric = m_metrics["base"]
        cur_metric["sent_n"] += 1
        if len(base_ops) == 0:
            cur_metric["sent_correct"] += 1
        _d, _s, _i = count_ops(base_ops)
        cur_metric["d"] += _d
        cur_metric["s"] += _s
        cur_metric["i"] += _i
        cur_metric["n"] += len(lab_arr)
        m_metrics["base"] = cur_metric
        if args.ref_metric == "base":
            ref_cmp_res = 0 if len(base_ops) == 0 else 1

        #update struct
        cur_metric = m_metrics["struct"]
        cur_metric["sent_n"] += 1
        if cmp_res == 0:
            cur_metric["sent_correct"] += 1
        _d, _s, _i = (0, 0, 0)
        cur_metric["d"] += _d
        cur_metric["s"] += _s
        cur_metric["i"] += _i
        cur_metric["n"] += 1
        m_metrics["struct"] = cur_metric
        if args.ref_metric == "struct":
            ref_cmp_res = cmp_res

        # update struct single
        cur_metric = m_metrics["struct.single"]
        cur_metric["sent_n"] += single_n
        cur_metric["sent_correct"] += single_acc
        cur_metric["n"] += 1
        m_metrics["struct.single"] = cur_metric
        if args.ref_metric == "struct.single":
            ref_cmp_res = 0 if single_acc > 0 else 1

        # update struct line
        cur_metric = m_metrics["struct.line"]
        cur_metric["sent_n"] += 1
        cmp_line = 1
        if cmp_res == 0 and cmp_rec_global_text.replace(" ", "") == cmp_lab_global_text.replace(" ", ""):
            cur_metric["sent_correct"] += 1
            cmp_line = 0
        cur_metric["n"] += 1
        m_metrics["struct.line"] = cur_metric
        if args.ref_metric == "struct.line":
            ref_cmp_res = cmp_line

        if args.refLog is not None and os.path.exists(args.refLog) and data_key in ref_set:
            cur_metric = m_metrics["struct.ref"]
            cur_metric["sent_n"] += 1
            if cmp_res == 0:
                cur_metric["sent_correct"] += 1
            cur_metric["n"] += 1
            m_metrics["struct.ref"] = cur_metric
            if args.ref_metric == "struct.ref":
                ref_cmp_res = cmp_res

            cur_metric = m_metrics["struct.ref.single"]
            cur_metric["sent_n"] += single_n
            cur_metric["sent_correct"] += single_acc
            cur_metric["n"] += 1
            m_metrics["struct.ref.single"] = cur_metric
            if args.ref_metric == "struct.ref.single":
                ref_cmp_res = 0 if single_acc > 0 else 1

            cur_metric = m_metrics["struct.ref.line"]
            cur_metric["sent_n"] += 1
            cmp_line = 1
            if cmp_res == 0 and cmp_rec_global_text.replace(" ", "") == cmp_lab_global_text.replace(" ", ""):
                cur_metric["sent_correct"] += 1
                cmp_line = 0
            cur_metric["n"] += 1
            m_metrics["struct.ref.line"] = cur_metric
            if args.ref_metric == "struct.ref.line":
                ref_cmp_res = cmp_line


        

    if m_err_lines_lock is not None:
        cur_out_line = line
        if ori_trans is not None:
            cur_out_line = "{}\t{}\t{}\t{}\n".format(img_path, ref_cmp_res, ori_trans, recs[0])
        else:
            cur_out_line = "{}\t{}\t{}\t{}\n".format(img_path, ref_cmp_res, lab, recs[0])
        with m_err_lines_lock:
            m_err_lines.put(cur_out_line)

    # if m_exception_lines_lock is not None and is_simple is True:
    #     with m_exception_lines_lock:
    #         m_exception_lines.put(line)

    records_queue.put(1)

def try_do_single_task(line, line_id, args, ref_set, \
        m_metrics, m_metrics_lock, \
        records_queue, records_queue_lock, \
        m_err_lines = None, m_err_lines_lock = None, \
        m_exception_lines = None, m_exception_lines_lock = None, \
        m_post_lines = None, m_post_lines_lock = None
    ):
    try:
        do_single_task(line, line_id, args, ref_set, \
            m_metrics, m_metrics_lock, \
            records_queue, records_queue_lock, \
            m_err_lines, m_err_lines_lock, \
            m_exception_lines, m_exception_lines_lock, \
            m_post_lines, m_post_lines_lock
        )
    except BaseException as e:
        err_string = "[exception] line_id={}\terror={}\tline={}".format(line_id, e, line.strip())
        print(err_string)
        with m_exception_lines_lock:
            m_exception_lines.put(err_string+"\n")
        if records_queue_lock is not None:
            with records_queue_lock:
                records_queue.put(1)
        #pdb.set_trace()

def main(args):
    args.origin_trans_dict = None
    if args.origin_trans is not None and os.path.exists(args.origin_trans):
        with open(args.origin_trans, "r") as fin:
            ori_lines = fin.readlines()
        args.origin_trans_dict = {}
        for ori_line in ori_lines:
            img_path, trans = ori_line.strip().split("\t", 1)
            img_key = os.path.splitext(os.path.basename(img_path))[0]
            args.origin_trans_dict[img_key] = trans

    ref_list = []
    if args.refLog is not None and os.path.exists(args.refLog):
        with open(args.refLog, "r") as fin:
            ref_lines = fin.readlines()
        for line in ref_lines:
            img_path = line.split("\t")[0]
            ref_key = os.path.splitext(os.path.basename(img_path))[0]
            ref_list.append(ref_key)
            # ref_list.append(line.split("\t")[0])
    else:
        args.refLog = None
    ref_set = set(ref_list)

    name_dict = {}
    with open(args.inLog, "r") as fin:
        lines = fin.readlines()
    for line in lines:
        key = line.split("\t")[0]
        key = os.path.splitext(os.path.basename(key))[0]
        name_dict[key] = line

    # pdb.set_trace()
    miss_refKey_cnt = 0
    for ref_key in ref_set:
        if ref_key not in name_dict:
            miss_refKey_cnt += 1
            #warnings.warn("ref key {} not in input log file".format(ref_key))
        # assert ref_key in name_dict
    if miss_refKey_cnt > 0:
        warnings.warn("{} ref keys not in input log file, total {} lines in ref, {} lines in log".format(miss_refKey_cnt, len(ref_set), len(name_dict)))

    # init output viz dir
    viz_dir = args.output
    if viz_dir is None or viz_dir == "":
        prefix, _ = os.path.splitext(args.inLog)
        viz_dir = "{}_viz".format(prefix)
    if not os.path.exists(viz_dir):
        if args.viz > 0:
            os.makedirs(viz_dir)
    args.output = viz_dir

    # init metrics
    manager = Manager()
    base_item = {"sent_n": 0, "sent_correct": 0, "d": 0, "s": 0, "i": 0, "n": 0}
    base_item = manager.dict(base_item)
    metrics = {"base": base_item.copy()}
    metrics["struct"] = base_item.copy()
    metrics["chem_no_angle"] = base_item.copy()
    metrics["struct.single"] = base_item.copy()
    metrics["struct.line"] = base_item.copy()
    if args.refLog is not None and os.path.exists(args.refLog):
        metrics["struct.ref"] = base_item.copy()
        metrics["struct.ref.single"] = base_item.copy()
        metrics["struct.ref.line"] = base_item.copy()

    #gen tasks
    records_queue = manager.Queue()
    records_queue_lock = manager.Lock()
    m_metrics = manager.dict(metrics)
    m_metrics_lock = manager.Lock()

    m_exception_lines = manager.Queue()
    m_exception_lines_lock = manager.Lock()

    if args.structErrPath is not None and args.structErrPath != "":
        if args.structErrPath == "auto":
            prefix, ext = os.path.splitext(args.inLog)
            args.structErrPath = "{}_metric[{}].err{}".format(prefix, args.ref_metric, ext)
        m_err_lines = manager.Queue()
        m_err_lines_lock = manager.Lock()
    else:
        m_err_lines = None
        m_err_lines_lock = None
    if args.post_log_path == "auto":
        prefix, ext = os.path.splitext(args.inLog)
        args.post_log_path = prefix + ".post" + ext
    if args.post_log_path is not None:
        m_post_lines = manager.Queue()
        m_post_lines_lock = manager.Lock()
    else:
        m_post_lines = None
        m_post_lines_lock = None

    all_tasks = []
    line_id = -1
    if args.num_workers <= 0:
        for line in tqdm.tqdm(lines):
            line_id += 1
            cur_task = (
                line, line_id, args, ref_set, m_metrics, m_metrics_lock, records_queue, records_queue_lock, m_err_lines, m_err_lines_lock, m_exception_lines, m_exception_lines_lock,
                m_post_lines, m_post_lines_lock
            )
            # if line.find("00123_0370") == -1:
            #     continue
            do_single_task(*cur_task)
    else:
        for line in tqdm.tqdm(lines):
            line_id += 1
            cur_task = (
                line, line_id, args, ref_set, m_metrics, m_metrics_lock, records_queue, records_queue_lock, m_err_lines, m_err_lines_lock, m_exception_lines, m_exception_lines_lock,
                m_post_lines, m_post_lines_lock
            )
            all_tasks.append(cur_task)
            pass

        def print_error(error):
            print("error:", error)

        poolSize = args.num_workers
        # pool = Pool(poolSize, initializer=init, initargs=(lrc_parser, ))
        pool = Pool(poolSize)
        pool.starmap_async(try_do_single_task, all_tasks, error_callback=print_error)
        pool.close()
        tq = tqdm.tqdm(total=len(all_tasks))
        count = 0
        print("begin")
        #try:
        while count < len(all_tasks):
            try:
                c = records_queue.get_nowait()
            except queue.Empty:
                continue
            count += 1
            tq.update(1)
        # except:
        #     pass
        pool.join()

    # pdb.set_trace()
    for metric_name in m_metrics:
        if metric_name == "chem_no_angle":
            continue
        metric = m_metrics[metric_name]
        sent_correct = metric["sent_correct"]
        sent_n = metric["sent_n"]
        d = metric["d"]
        s = metric["s"]
        i = metric["i"]
        n = metric["n"]
        if sent_n == 0:
            sent_n = 1e-10
        if n == 0:
            n = 1e-10
        sent_acc = float(sent_correct) / (sent_n)
        word_acc = float(n - d - s - i) / (n)
        word_cor = float(n - d - s) / (n)
        print("------ metric {} ------".format(metric_name))
        print("wacc={:.4f} % wcor={:.4f} % d={} s={} i={} n={}".format(word_acc * 100.0, word_cor * 100.0, d, s, i, n))
        print("sent acc = {:.4f} %( {}/{} )".format(sent_acc * 100.0, sent_correct, sent_n))

    if args.structErrPath is not None and args.structErrPath != "":
        with open(args.structErrPath, "w") as fout:
            while not m_err_lines.empty():
                line = m_err_lines.get()
                fout.write(line)

    if args.post_log_path is not None:
        with open(args.post_log_path, "w") as fout:
            while not m_post_lines.empty():
                line = m_post_lines.get()
                fout.write(line)
    
    if not m_exception_lines.empty():
        prefix, ext = os.path.splitext(args.inLog)
        exception_path = "{}.exception.{}".format(prefix, args.ref_metric, ext)
        with open(exception_path, "w") as fout:
            while not m_exception_lines.empty():
                line = m_exception_lines.get()
                fout.write(line)

    pass
