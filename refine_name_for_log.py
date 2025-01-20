from typing import OrderedDict
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import Levenshtein
import sys
import random
import tqdm
import pdb
import argparse
from multiprocessing import Process, synchronize, Lock, Manager, Pool
from six.moves import queue
import sys
from data_encapsulation import ListRecordLoader


def load_lab_rec(in_log):
    lab_dict = OrderedDict()
    rec_dict = OrderedDict()
    print('Loading label and rec ...')
    with open(in_log, "r") as lff:
        for line in tqdm.tqdm(lff):
            segs = line.strip().split('\t')
            if len(segs) < 1:
                print('Error Line: %s' % line)
                continue
            name = segs[0]
            img_path = name
            lab_dict[img_path] = segs[1] if len(segs) >= 2 else ""
            rec_dict[img_path] = segs[2] if len(segs) >= 3 else ""
    print('Get Valid label %d' % len(lab_dict))
    print('Get Valid pred %d' % len(rec_dict))
    return lab_dict, rec_dict

def main(args):
    #load lrc
    print("load lrc...")
    sdr = ListRecordLoader(args.lrc_path)
    
    #load lab and rec
    print("load lines")
    #lab_dict, rec_dict = load_lab_rec(args.input)
    with open(args.input, "r") as fin:
        lines = fin.readlines()

    output = args.output
    # pdb.set_trace()
    if output is None:
        prefix, ext = os.path.splitext(args.input)
        # pdb.set_trace()
        output = "{}_wName{}".format(prefix, ext)

    new_lines = []
    for line in tqdm.tqdm(lines):
        spts = line.split("\t")
        key = spts[0]
        record_idx, *idxes = [int(item) for item in key.split('-')]
        record = sdr.get_record(record_idx)
        image_path = record[args.key]
        if args.addPath > 0:
            image_name = image_path
        else:
            image_name = os.path.basename(image_path)
        new_key = image_name + "-" + "-".join(["%d"%idx for idx in idxes])
        new_line = "\t".join([new_key]+ spts[1:])
        new_lines.append(new_line)

    new_lines = sorted(new_lines, key = lambda x:x.split("\t")[0])
    with open(output, "w") as fout:
        fout.writelines(new_lines)
    
    print('All Done!')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default="")
    parser.add_argument("-lrc_path", type=str, default="")
    parser.add_argument("-addPath", type=int, default=0, help="")
    parser.add_argument("-key", type=str, default="image_path")
    parser.add_argument("-output", type=str, default=None)

    args = parser.parse_args()
    main(args)

