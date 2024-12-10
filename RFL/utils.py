import os, sys
import cv2
import math
import numpy
import pdb
from collections import OrderedDict
import re
import Levenshtein


pair_dict = {"}":"{", "]":"["}
def replace_chemfig(text):
    replace_dict = OrderedDict()
    ind = 0
    new_text = ""
    while True:
        pos = text.find("\\chemfig")
        if pos == -1:
            break
        cur_pos = pos + 8
        cur_left_pair = None
        cur_left_pos = None
        curLevel = 0
        range_cnt = {"[":0, "{":0}
        while cur_pos < len(text):
            ch = text[cur_pos]
            if ch == "[" or ch == "{":
                if cur_left_pair is None:
                    cur_left_pair = ch
                    cur_left_pos = cur_pos
                    curLevel = 1
                elif cur_left_pair == ch:
                    curLevel += 1
            elif ch == "}" or ch == "]":
                if cur_left_pair == pair_dict[ch]:
                    curLevel -= 1
                    if curLevel == 0:
                        range_cnt[cur_left_pair] += 1
                        if range_cnt["["] > 1:
                            raise ValueError("multiple attr range")
                        if range_cnt["{"] >= 1:
                            # pdb.set_trace()
                            break
                        cur_left_pair = None
                        cur_left_pos = None
                    
            elif cur_left_pair is None:
                if ch != " ":
                    raise ValueError("format err, input = {}".format(text))
            else:
                pass
            cur_pos += 1
        beginPos = cur_left_pos
        endPos = cur_pos + 1
        rep_key = "\\chem{}".format(chr(ord('a') + ind))
        ind += 1
        replace_dict[rep_key] = "\\chemfig "+text[beginPos:endPos]
        text = text[0:pos] + " " + rep_key + " " + text[endPos:]
        new_text += replace_dict[rep_key] + " "

        pos = cur_pos + 1
    return new_text, replace_dict, text


def get_atom_group(item_list):
    '''检测原子团，并将item_list中分开的原子团替换为合并之后的原子团'''
    lengths = [len(item) for item in item_list if isinstance(item, str)]
    # print(lengths)
    # print(item_list)
    consecutive_ones = []
    start_index = None
    end_index = None
    for i, num in enumerate(lengths):
        if i == 0 or i == len(item_list)-1:
            continue # 跳过第一个和最后一个 { }
        if num == 1:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                end_index = i - 1
                if end_index - start_index + 1 >= 2:
                    consecutive_ones.append((start_index, end_index))
                start_index = None
    
    if start_index is not None:
        end_index = len(lengths) - 2
        if end_index - start_index + 1 >= 2:
            consecutive_ones.append((start_index, end_index))

    for item in reversed(consecutive_ones):
        # replace
        # print(item[0], item[1])
        atom_group = "".join(item_list[item[0]: item[1]+1])
        print(atom_group)
        del item_list[item[0]: item[1] + 1]
        # item_list[item[0]: item[1]] = atom_group
        item_list.insert(item[0], atom_group)
    
    # print(item_list)
    return item_list
    # print(lengths[consecutive_ones])

def scan_dir(in_dir, ext, rescan=False):
    cache_path = os.path.join(in_dir, "{}.cache.txt".format(ext))
    if not os.path.exists(cache_path) or rescan is True:
        cmd = "find {} -name \*.{}".format(in_dir, ext)
        in_lines = os.popen(cmd).readlines()
        with open(cache_path, "w") as fout:
            fout.writelines(in_lines)
    else:
        with open(cache_path, "r") as fin:
            in_lines = fin.readlines()
    return in_lines

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

    out_text_arr = tmp_text_arr
    return out_text_arr

def removeAngle(inArr):
    outArr = []
    for unit in inArr:
        # pos = unit.find("[")
        # if pos != -1 and unit.endswith("]"):
        #     pdb.set_trace()
        #     outArr.append(unit[:pos])
        # else:
        #     outArr.append(unit)
        tgt_unit = re.sub(r"\[:[0-9]*\]","", unit) #fixed at 2022.03.04 for bug in ?[a,{=}]
        outArr.append(unit)
    return outArr

#====================== text process =======================#

def norm_sub(in_text_arr):
    out_text_arr = []
    # for i, text in enumerate(tmp_text_arr):
    ind = 0
    left_start = False
    while ind < len(in_text_arr):
        if in_text_arr[ind] == "_" and ind + 1 < len(in_text_arr) and in_text_arr[ind + 1] == "{":
            left_start = True
            ind += 2
            continue
        if left_start and in_text_arr[ind] == "}":
            left_start = False
            ind += 1
            continue
        out_text_arr.append(in_text_arr[ind])
        ind += 1
    return out_text_arr

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
        out_text_arr = norm_sub(tmp_text_arr)
    else:
        out_text_arr = tmp_text_arr
    return out_text_arr

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


def post_process(input_arr):
    output_arr = input_arr
    output_arr = [temp for temp in output_arr if temp != "\\smear" and temp != "\\space"]
    output_arr = rm_underline_textbf(output_arr)
    return output_arr

#====================process for reverser texlive render==============#


null_str=set(["\t", " ", "\u00A0", "\u3000"])
def rm_bracket_v2(inStr, prefixs = ["\\textit"]):
    curStr = inStr
    # curStr = list(filter(None, curStr))
    for cur_prefix in prefixs:
        pos = -1
        while True:
            pos = curStr.find(cur_prefix, pos+1)
            if pos == -1:
                break
            prefix_end = pos + len(cur_prefix)
            pos_left = curStr.find("{", prefix_end)
            if pos_left == -1:
                continue
            valid = True
            for ind in range(prefix_end, pos_left):
                if curStr[ind] not in null_str:
                    valid = False
                    break
            if not valid:
                continue
            ind = pos_left + 1
            pos_right = -1
            curLevel =1
            while ind < len(curStr):
                if curStr[ind] == "{":
                    curLevel += 1
                elif curStr[ind] == "}":
                    curLevel -= 1
                    if curLevel == 0:
                        pos_right = ind
                        break
                ind += 1
            curStr = curStr[0:pos] + curStr[pos_left+1:pos_right] + curStr[pos_right+1:]
    return curStr

def IsChinese(uni_num):
    if uni_num >= 0x4E00 and uni_num <= 0x9FBF:
        return True
    elif uni_num>=0xF900 and uni_num <= 0xFAFF:
        return True
    else:
        return False

def process_trans_for_texlive(trans):
    # rm textit
    trans = rm_bracket_v2(trans)
    # process chinese
    new_trans = ""
    is_chn_range = False
    for ch in trans:
        if len(ch) != 1:
            continue
        uni_num = ord(ch)
        if IsChinese(uni_num) and not is_chn_range:
            new_trans += "\\text{"
            is_chn_range = True
        if not IsChinese(uni_num) and is_chn_range:
            new_trans += "}"
            is_chn_range = False
        new_trans += ch
    if is_chn_range:
        new_trans += "}"
        is_chn_range = False
    # new_trans = "$" + new_trans + "$"

    new_trans = new_trans.replace("\r\n", "\\\\")
    new_trans = new_trans.replace("\r", "\\\\")
    new_trans = new_trans.replace("\n", "\\\\")
    new_trans = new_trans.replace("\\smear", "")
    new_trans = new_trans.replace("\\enter", "\\\\")
    new_trans = new_trans.replace("\\space", "\\quad")
    new_trans = new_trans.replace("\\unk", "")

    # process "\\"
    spts = new_trans.split("\\\\")
    new_trans = "\\\\".join(["$ \\rm "+spt+"$" for spt in spts if len(spt)>0])
    
    return new_trans
