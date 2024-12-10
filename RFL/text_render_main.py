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

_support_format = set(["text", "lrc", "xml"])

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
        out_text = text_render.text_render(in_text, debug=False, branch_de_amb=1)
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
        area = record["sub_areas"]
        top_id = 0
        areas_arr = [x for x in area]
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
        #shutil.copy(record["image_path"], "./debug/origin.jpg")
        out_records = shared_params["out_records"]
        out_records_lock = shared_params["out_records_lock"]
        with out_records_lock:
            out_records.put(record)
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
    
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default=None)
    parser.add_argument("-output", type=str, default=None)
    parser.add_argument("-input_type", type=str, default="text", help="current support {}".format(_support_format))
    parser.add_argument("-use_raw_info", type=int, default=0, help="for lrc mode")
    parser.add_argument("-num_workers", type=int, default=32)
    args = parser.parse_args()
    main(args)
