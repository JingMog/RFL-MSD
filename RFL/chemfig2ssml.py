from multiprocessing import Process, synchronize, Lock, Manager, Pool
from tqdm import tqdm
from six.moves import queue
import argparse

from cond_render_main import process_cond_render
from text_render import rend_text
from chemfig_struct import *

def chemfig_convert_to_ssml(input_chemfig):
    out_units = process_cond_render(input_chemfig)
    preprocess_chemfig = []
    for id, unit in enumerate(out_units):
        if not isinstance(unit, Atom):
            # 非分子部分直接并入out_units
            preprocess_chemfig.append(unit)
        else:
            # 化学分子
            temp_string = rend_text(unit)
            temp_string = ['\chemfig', '{'] + temp_string + ['}']
            preprocess_chemfig += temp_string
    preprocess_chemfig = " ".join(preprocess_chemfig)
    return preprocess_chemfig


def do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    
    line = line.strip().split('\t')
    file_path = line[0]
    input_chemfig = line[1]
    ssml_string = chemfig_convert_to_ssml(input_chemfig)

    out_cs_string = shared_params["out_cs_string"]
    out_cs_string_lock = shared_params["out_cs_string_lock"]
    with out_cs_string_lock:
        out_cs_string.put([file_path, ssml_string])


def try_do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    try:
        do_single_task(params, args, records_queue, records_queue_lock, shared_params)
    except BaseException as e:
        error_lines = shared_params["error_lines"]
        error_lines_lock = shared_params["error_lines_lock"]
        with error_lines_lock:
            file_name = params["line"].strip().split('\t')[0]
            label_string = params["line"].strip().split('\t')[0]
            error_lines.put(file_name + '\t' + str(e) + '\t' + label_string + '\n')
        # print("try fail!")
    if records_queue_lock is not None:
        with records_queue_lock:
            records_queue.put(1)

def main(args):
    if args.input_type == "text":
        with open(args.input, "r") as fin:
            lines = fin.readlines()
    else:
        raise NotImplementedError("unsupport input type = {}".format(args.input_type))
    
    # init metrics
    manager = Manager()
    records_queue = manager.Queue()
    records_queue_lock = manager.Lock()

    shared_params = {}
    if args.input_type == "text":
        shared_params["error_lines"] = manager.Queue()
        shared_params["error_lines_lock"] = manager.Lock()
        shared_params["out_cs_string"] = manager.Queue()
        shared_params["out_cs_string_lock"] = manager.Lock()

    
    all_tasks = []
    line_id = -1
    
    if args.num_workers <= 0:
        for line in tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id
            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            do_single_task(*cur_task)
            if line_id > 20:
                break
    else:
        for line in tqdm(lines):
            line_id += 1
            params = {}
            params["line"] = line
            params["line_id"] = line_id
            cur_task = (params, args, records_queue, records_queue_lock, shared_params)
            all_tasks.append(cur_task)
            # if line_id > 100:
            #     break
            
        def print_error(error):
            print("error:", error)

        poolSize = args.num_workers
        pool = Pool(poolSize)
        pool.starmap_async(try_do_single_task, all_tasks, error_callback=print_error)
        pool.close()
        tq = tqdm(total=len(all_tasks))
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

        pool.join()

    # 后处理保存转换错误的结果和生成的骨干字符串
    error_lines = shared_params["error_lines"]
    error_lines_lock = shared_params["error_lines_lock"]
    # print(error_lines)
    with open(args.error_output, "w") as fout:
        while not error_lines.empty():
            line = error_lines.get()
            fout.write(line)
    
    out_cs_string = shared_params["out_cs_string"]
    out_cs_string_lock = shared_params["out_cs_string_lock"]
    with open(args.output, 'w') as fout:
        while not out_cs_string.empty():
            line = out_cs_string.get()
            file_name = line[0]
            tmp_ssml_string = line[1]
            fout.write(file_name + '\t' + tmp_ssml_string + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default='chemfig_test.txt')
    parser.add_argument("-output", type=str, default='ssml_test.txt')
    parser.add_argument("-error_output", type=str, default='ssml_test_error.txt')
    parser.add_argument("-input_type", type=str, default="text", help="current support text")
    parser.add_argument("-num_workers", type=int, default=40)
    args = parser.parse_args()
    main(args)

