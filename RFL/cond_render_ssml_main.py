import os, sys
import pdb
import argparse
import tqdm
from multiprocessing import Process, synchronize, Lock, Manager, Pool
import multiprocessing
from six.moves import queue
import text_render
import utils
import shutil
import cond_render as chemfig_cond_render
import pickle
from Tool_Formula.latex_norm.transcription import parse_transcription
from chemfig_struct import *

_support_format = set(["text", "lrc", "xml"])

def process_cond_render(text):
    chemfig_text, rep_dict, rep_text = utils.replace_chemfig(text)
    rep_text_units = parse_transcription(rep_text, simple_trans=True)
    new_rep_dict = {}
    for key, value in rep_dict.items():
        # chemfig_str = "".join(list(filter(lambda x:x!=" ", value)))
        chemfig_str = value.strip()[8:].strip()[1:-1]
        try:
            rep_atom = chemfig_cond_render.chemfig_random_cond_parse(chemfig_str)
            new_rep_dict[key] = rep_atom
        except Exception as e:
            new_rep_dict[key] = value
            raise ValueError("Parse Error, err = {}".format(e))
    out_units = []
    for text_unit in rep_text_units:
        if text_unit in new_rep_dict:
            out_units.append(new_rep_dict[text_unit])
        else:
            out_units.append(text_unit)
    return out_units

def do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    if args.input_type == "text":
        out_lines = shared_params["out_lines"]
        out_lines_lock = shared_params["out_lines_lock"]

        spts= line.strip().split("\t")
        if len(spts) != 2:
            return
        img_key, in_text = spts
        out_text = text_render.text_render(in_text, debug=False)
        with out_lines_lock:
            out_lines.put("{}\t{}\n".format(img_key, out_text))
    elif args.input_type == "xml":
        pass
    elif args.input_type == "lrc":
        if "lrc_parser" in params:
            lrc_parser = params["lrc_parser"]
        else:
            lrc_parser = _lrc_parser
        idx = line
        record = lrc_parser.get_record(idx)
        
        # ============= for cond render ===============
        area_queue = [(x, []) for x in record["sub_areas"]]
        success = True
        while len(area_queue):
            cur_area, prefix_arr = area_queue.pop(0)
            cur_prefix_arr = prefix_arr + [cur_area["idx"]]
            cur_idx = "-".join(cur_prefix_arr)
            if "sub_areas" in cur_area:
                area_queue += [(x, cur_prefix_arr) for x in cur_area["sub_areas"]]
            if "text" in cur_area:
                if args.use_raw_info > 0:
                    if "raw_info" in cur_area and "text" in cur_area["raw_info"]:
                        input_text = cur_area["raw_info"]["text"]
                    else:
                        raise ValueError("can not find raw info")
                else:
                    input_text = cur_area["text"]
                

                out_text_units = process_cond_render(input_text) ######
                unit_bytes = pickle.dumps(out_text_units)
                if "raw_info" not in cur_area:
                    warnings.warn("can not find raw_info")
                cur_area["raw_info"]["struct_text"] = unit_bytes
                #for test dump
                if args.rend_check > 0:
                    # out_text = cur_area["text"]
                    try_cnt = 0
                    while True:
                        try:
                            out_parsed_units = chemfig_cond_render.process_text_rnd_cond_render(out_text_units)
                            out_text = " ".join([x[0] for x in out_parsed_units])
                            break
                        except BaseException as e:
                            if try_cnt < 5:
                                try_cnt += 1
                                continue
                            else:
                                raise ValueError(e)


                        
                    #cur_area["text"] = out_text

                # except BaseException as e:
                #     success = False
                #     print("can not write, check record={} err={}".format(idx, e))
        # ============= for cond render ===============
        #pdb.set_trace()
        # ==============for text render ===============
        if success:
            top_id = 0
            areas_arr = [x for x in record["sub_areas"]]
            while top_id < len(areas_arr):
                cur_area = areas_arr[top_id]
                top_id += 1
                if "text" in cur_area:
                    if args.use_raw_info > 0:
                        if "raw_info" in cur_area and "text" in cur_area["raw_info"]:
                            in_text = cur_area["raw_info"]["text"]
                        else:
                            raise ValueError("can not find raw info")
                    else:
                        in_text = cur_area["text"]
                    cur_area["text"] = text_render.text_render(in_text, debug=False, branch_de_amb=1)
                if "sub_areas" in cur_area:
                    areas_arr += cur_area["sub_areas"]
        # ==============for text render ===============

        # shutil.copy(record["image_path"], "./debug/origin.jpg")
        if success:
            out_records = shared_params["out_records"]
            out_records_lock = shared_params["out_records_lock"]
            with out_records_lock:
                out_records.put(record)
        # pdb.set_trace()
        pass
    
    pass


def try_do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    try:
        do_single_task(params, args, records_queue, records_queue_lock, shared_params)
    except BaseException as e:
        line_content = line if type(line) is str else line
        print("try fail! line id = {}	line = {} err= {}".format(line_id, line_content, e))
    if records_queue_lock is not None:
        with records_queue_lock:
            records_queue.put(1)


def main(args):
    if args.input_type == "text":
        with open(args.input, "r") as fin:
            lines = fin.readlines()
    elif args.input_type == "xml":
        lines = utils.scan_dir(args.input, "xml")
    else:
        raise NotImplementedError("unsupport input type = {}".format(args.input_type))
    
        
    # init metrics
    manager = Manager()
    records_queue = manager.Queue()
    records_queue_lock = manager.Lock()

    # TODO add share params here
    shared_params = {}
    if args.input_type == "text":
        shared_params["out_lines"] = manager.Queue()
        shared_params["out_lines_lock"] = manager.Lock()
    elif args.input_type == "xml":
        common_prefix = os.path.commonpath(lines)
        # treat args.output as output_dir
    # shared_params["err_lines"] = manager.Queue()
    # shared_params["err_lines_lock"] = manager.Lock()

    all_tasks = []
    line_id = -1
    
    if args.num_workers <= 0:
        for line in tqdm.tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id
            if args.input_type == "xml":
                params["common_prefix"] = common_prefix
            if args.input_type == "lrc":
                params["lrc_parser"] = lrc_parser
            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            do_single_task(*cur_task)
            if line_id > 20:
                break
            #pdb.set_trace()
    else:
        for line in tqdm.tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id
            if args.input_type == "xml":
                params["common_prefix"] = common_prefix
            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            all_tasks.append(cur_task)
            pass

        def print_error(error):
            print("error:", error)

        def init(a):
            global _lrc_parser
            _lrc_parser = a

        poolSize = args.num_workers
        if args.input_type == "lrc":
            pool = Pool(poolSize, initializer=init, initargs=(lrc_parser, ))
        else:
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
            if args.input_type == "lrc":
                try:
                    with shared_params["out_records_lock"]:
                        cur_record = shared_params["out_records"].get_nowait()
                    lrc_writer.add_record(cur_record)
                except queue.Empty:
                    pass
            count += 1
            tq.update(1)
        # except:
        #     pass
        pool.join()

    #TODO add  post process
    if args.input_type == "text":
        out_lines = shared_params["out_lines"]
        out_lines_lock = shared_params["out_lines_lock"]
        with open(args.output, "w") as fout:
            while not out_lines.empty():
                line = out_lines.get()
                fout.write(line)
    elif args.input_type == "xml":
        pass
        # treat args.output as output_dir
        # treat args.output as output_lrc_path
    
    pass

