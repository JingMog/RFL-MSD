import os, sys
import argparse
import pdb
import tqdm
from RFL_ import utils

def main(args):
    with open(args.input, "r") as fin:
        lines = fin.readlines()

    output = args.output
    if output is None:
        prefix, ext = os.path.splitext(args.input)
        output = "{}_chemprocess{}".format(prefix, ext)
    
    fout = open(output, "w")

    for ind, line in enumerate(tqdm.tqdm(lines)):
        spts =line.strip().split("\t")
        if len(spts) == 2:
            spts.append("")
        if len(spts) !=3:
            print(line)
            continue
        img_key, lab, rec = spts
        _, rep_dict, remain_trans = utils.replace_chemfig(rec)
        for key, trans in rep_dict.items():
            words = trans.split(" ")
            new_words = []
            for word in words:
                if word in ["(", ")", "-"]:
                    word = "{"+word+"}"
                word = word.replace("branch", "")
                new_words.append(word)
            rep_dict[key] = " ".join(new_words)
        #pdb.set_trace()
        remain_spts = remain_trans.split(" ")
        out_spts = []
        for remain_spt in remain_spts:
            if remain_spt in rep_dict:
                out_spts.append(rep_dict[remain_spt])
            else:
                out_spts.append(remain_spt)
        # for key, new_trans in rep_dict.items():
        #     remain_trans = remain_trans.replace(key, new_trans)
        out_rec = " ".join(out_spts)
        out_rec = out_rec.replace('branch', '')
        fout.write("{}\t{}\t{}\n".format(img_key, lab, out_rec))
    fout.close()
    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str)
    parser.add_argument("-output", type=str, default=None)
    parser.add_argument("-num_workers", type=int, default=32)
    args = parser.parse_args()
    main(args)
