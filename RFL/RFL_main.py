# @Time      : 2024/3/11

from RFL.RFL import *
import argparse
import pickle
from multiprocessing import Process, synchronize, Lock, Manager, Pool
from six.moves import queue


def do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    line = params["line"]
    line_id = params["line_id"]
    
    if args.input_type == "text":
        line = line.strip().split('\t')
        file_path = line[0]
        input_chemfig = line[1]
        success, cs_string, _, ring_branch_info, cond_data = cs_main(input_chemfig, is_show=False)

        if success:
            out_cs_string = shared_params["out_cs_string"]
            out_cs_string_lock = shared_params["out_cs_string_lock"]
            with out_cs_string_lock:
                cs_string = " ".join(cs_string)
                out_cs_string.put([file_path, cs_string, ring_branch_info, cond_data])
        else:
            error_lines = shared_params["error_lines"]
            error_lines_lock = shared_params["error_lines_lock"]
            with error_lines_lock:
                error_lines.put(params["line"])

def try_do_single_task(params, args, records_queue, records_queue_lock, shared_params={}):
    try:
        do_single_task(params, args, records_queue, records_queue_lock, shared_params)
    except BaseException as e:
        error_lines = shared_params["error_lines"]
        error_lines_lock = shared_params["error_lines_lock"]
        with error_lines_lock:
            file_name = params["line"].strip().split('\t')[0]
            label_string = params["line"].strip().split('\t')[1]
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
        shared_params["out_branch_info"] = manager.Queue()
        shared_params["out_branch_info_lock"] = manager.Lock()

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
    
    if args.input_type == "text":
        out_cs_string = shared_params["out_cs_string"]
        out_cs_string_lock = shared_params["out_cs_string_lock"]
        max_len = 1
        with open(args.output, 'w') as fout:
            while not out_cs_string.empty():
                line = out_cs_string.get()
                file_name = line[0]
                tmp_cs_string = line[1]
                ring_branch_info = line[2]
                cond_data = line[3]
                cur_max_len = [len(item) for item in ring_branch_info if item is not None]
                if len(cur_max_len) > 0:
                    max_len = max(max_len, max(cur_max_len))

                fout.write(file_name + '\t' + tmp_cs_string + '\n')
        # print("ring_branch_info最大长度: ", max_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default='valid_ssml_sd.txt')
    parser.add_argument("-output", type=str, default='./result/valid_RFL_string.txt')
    parser.add_argument("-error_output", type=str, default='./result/error_example.txt')
    parser.add_argument("-input_type", type=str, default="text", help="current support text")
    parser.add_argument("-num_workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
    
