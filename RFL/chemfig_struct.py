import os, sys
import re
import pdb
import numpy as np
import math

import Tool_Formula.latex_norm.transcription as transcription
from Tool_Formula.latex_norm.transcription import parse_transcription
from Tool_Formula.latex_norm.reverse_transcription import reverse_transcription
from Tool_Formula.latex_norm.transcription import Node
import warnings
import copy

bond_types = ["-", "=", "~", ">", "<", ">:", "<:", ">|", "<|", "-:", "=_", "=^", "~/", "-@"]
super_bond_types = ["-@", "=@", "~@", ">@", "<@", ">:@", "<:@", ">|@", "<|@", "=_@", "=^@", "~/@",
                    "-#", "=#", "~#", ">#", "<#", ">:#", "<:#", ">|#", "<|#", "=_#", "=^#", "~/#"]
bond_types = bond_types + super_bond_types
bond_types = sorted(bond_types, key=lambda x: -len(x))

bond_in_out_dict = {"-": "-", "=": "=", "~": "\equiv", "-:": ""}

special_chars = ["[", "]", "(", ")", "?", "\Charge", "\Chemabove", "\Chembelow", "*", "\mcfcringle"]
special_chars = set(special_chars)

escape_strs = [
    "\Charge", "\Chemabove", "\Chembelow", "\\", "\[", "\]", "\(", "\)", "\?", "{(}", "{)}", "\Circle", "\circle", "\mcfcringle"
]  #prevent normal text be treated as specail
escape_strs = sorted(escape_strs, key=lambda x: -len(x))
Node._Node__bin_ops += ["\\Chemabove", "\\Chembelow", "\\chemabove", "\\chembelow"]
Node._Node__uni_ops += ["\\phantom"]
transcription.rep_dict["\\mcfminus"] = "-"
transcription.rep_dict["\\mcfplus"] = "+"
transcription.rep_dict["\\mcfright"] = ""

directed_bond_types = {
    ">": "<",
    "<": ">",
    ">:": "<:",
    "<:": ">:",
    ">|": "<|",
    "<|": ">|",
}


class Atom(object):
    index = 0

    def __init__(self, text=""):
        self.name = "Atom_{}".format(Atom.index)
        Atom.index += 1
        self.m_text = text
        self.pos_x = 0
        self.pos_y = 0
        self.in_bonds = []
        self.out_bonds = []
        self.start_hooks = []
        self.end_hooks = []
        self.ring_ids = {}
        self.ring_branch_info = [] # 保存环上原子与分支相连的信息
        pass
    
    def __repr__(self):
        out_str = "<Atom"
        out_str += " name={}".format(self.name)
        out_str += " text={}".format(self.m_text)
        out_str += " x={}".format(self.pos_x)
        out_str += " y={}".format(self.pos_y)
        out_str += ">"
        return out_str
    
    def norm_chem_above_below(self, text_arr):
        rootNode = transcription.parse_tree(text_arr)
        node_stack = [rootNode]
        cursor = 0
        while len(node_stack) > 0 and cursor < len(node_stack):
            curNode = node_stack[cursor]
            cursor += 1
            for word_id, word in enumerate(curNode.words):
                if type(word) is Node:
                    node_stack.append(word)
                elif word == "\\chemabove":
                    curNode.words[word_id] = "\\Chemabove"
                    pass
                elif word == "\\Chembelow" or word == "\\chembelow":
                    curNode.words[word_id] = "\\Chemabove"
                    # swap
                    first = curNode.words[word_id+1]
                    second = curNode.words[word_id+2]
                    curNode.words[word_id+1] = second
                    curNode.words[word_id+2] = first
        rootNode.fix_op_nest()
        return rootNode.flatten()

    def normed_text(self):
        # if self.m_text.find("hemabove") != -1:
        #     pdb.set_trace()
        if self.m_text is not None and self.m_text != "":
            text_arr = parse_transcription(self.m_text, simple_trans=True)
            text_arr = self.norm_chem_above_below(text_arr)
        else:
            text_arr = []
        for end_hook in self.end_hooks:
            hook_str = "?["
            hook_str += end_hook.m_hookname
            if end_hook.m_bondtype is not None:
                hook_str += ",{{{}}}".format(end_hook.m_bondtype)
            hook_str += "]"
            text_arr.append(hook_str)
        for start_hook in self.start_hooks:
            hook_str = "?["
            hook_str += start_hook.m_hookname
            # if start_hook.m_bondtype is not None:
            #     hook_str += ",{}".format(start_hook.m_bondtype)
            hook_str += "]"
            text_arr.append(hook_str)
        return text_arr

    def reverse_normed_text(self):
        if self.m_text is not None and self.m_text != "":
            # pdb.set_trace()
            # text_arr = parse_transcription(self.m_text, simple_trans=True)
            # if self.m_text.find("\\mcfplus") != -1:
            #     pdb.set_trace()
            # text_arr, bad = reverse_transcription(" ".join(parse_transcription(self.m_text, simple_trans=True)))  # modified 2022.05.18
            # if self.m_text.find("hemabove") != -1:
            #     pdb.set_trace()
            out_text, bad = reverse_transcription(self.m_text)  # modified 2022.05.18
            # text_arr = out_text.split(" ")
            # new_text_arr = []
            # for text in text_arr:
            #     if text in ["-"]:
            #         new_text_arr.append("{"+text+"}")
            #     else:
            #         new_text_arr.append(text)
            # out_text = " ".join(new_text_arr)
            text_arr = [out_text]
            if bad:
                text_arr = [] + self.m_text.split(" ")
        else:
            text_arr = []
        for end_hook in self.end_hooks:
            hook_str = "?["
            hook_str += end_hook.m_hookname
            if end_hook.m_bondtype is not None:
                hook_str += ",{{{}}}".format(end_hook.m_bondtype)
            hook_str += "]"
            text_arr.append(hook_str)
        for start_hook in self.start_hooks:
            hook_str = "?["
            hook_str += start_hook.m_hookname
            # if start_hook.m_bondtype is not None:
            #     hook_str += ",{}".format(start_hook.m_bondtype)
            hook_str += "]"
            text_arr.append(hook_str)
        return text_arr


class CircleAtom(Atom):
    def __init__(self, text=""):
        super(CircleAtom, self).__init__(text)
        self.m_radius = 1.0
        self.m_text = "\\circle"

    def __repr__(self):
        out_str = "<CircleAtom"
        out_str += " name={}".format(self.name)
        out_str += " text={}".format(self.m_text)
        out_str += " radius={}".format(self.m_radius)
        out_str += " x={}".format(self.pos_x)
        out_str += " y={}".format(self.pos_y)
        out_str += ">"
        return out_str


class Charge(Atom):
    def __init__(self, text=""):
        super(Charge, self).__init__(text)
        self.m_charges = []
        pass

    def parse_attr_from_str(self, charge_info, center_info):
        assert center_info.startswith("{") and center_info.endswith("}")
        assert charge_info.startswith("{") and charge_info.endswith("}")
        # parse center info
        self.m_text = center_info[1:-1].strip()  #fixed @ 2022.04.19
        # parse charge info
        charge_info_content = charge_info[1:-1]
        charge_info_content = "".join(list(filter(lambda x: x != " ", charge_info_content)))
        if len(re.findall("\[[0-9.]*\]", charge_info_content)) > 0:
            charge_str_list = re.split("\[|\]", charge_info_content)
            charge_str_list = list(filter(None, charge_str_list))
            charge_str_list = [charge_str_list[2 * x] + charge_str_list[2*x + 1] for x in range(len(charge_str_list) // 2)]
        else:
            charge_str_list = charge_info[1:-1].split(",")
        # pdb.set_trace()
        for charge_str in charge_str_list:
            if charge_str is None or charge_str == "":
                continue
            try:
                pos_str, charge_type = charge_str.split("=")
                pos_str = pos_str.strip()  #fixed @ 2022.04.19
                charge_type = charge_type.strip()  #fixed @ 2022.04.19
            except:
                raise ValueError("can parse charge str = %s" % charge_str)
            spts = pos_str.split(":")
            angle = float(spts[0]) % 360
            angle = int(angle) % 360  # fixed @ 2022.04.20
            distance = 0
            if len(spts) > 1:
                assert spts[1].endswith("pt")
                distance = int(float(spts[1][:-2]))
            self.m_charges.append({"charge": charge_type, "angle": angle, "distance": distance})
        self.m_charges = sorted(self.m_charges, key=lambda x: x["angle"] * 10000 + x["distance"])

    def __repr__(self):
        out_str = "<Charge"
        out_str += " name={}".format(self.name)
        out_str += " text={}".format(self.m_text)
        out_str += " x={}".format(self.pos_x)
        out_str += " y={}".format(self.pos_y)
        for charge in self.m_charges:
            out_str += " [{} {} {}]".format(charge["angle"], charge["distance"], charge["charge"])
        out_str += ">"
        return out_str

    def normed_text(self):
        text_arr = ["\\Charge", "{"]
        for charge in self.m_charges:
            text_arr.append("[{}]".format(int(charge["angle"]) % 360))
            if charge["distance"] > 0:
                text_arr.append(":")
                text_arr.append("{}pt".format(int(charge["distance"])))
            text_arr.append("=")
            #text_arr.append(charge["charge"])
            if charge["charge"] in ["\\:", "\\|", "\\.", "\\"]:
                text_arr.append(charge["charge"])
            else:
                text_arr += parse_transcription(charge["charge"], simple_trans=True)
        text_arr.append("}")
        text_arr.append("{")
        text_arr += parse_transcription(self.m_text, simple_trans=True)  #fixed 2022.04.19
        text_arr.append("}")
        return text_arr

    def reverse_normed_text(self):
        text_arr = ["\\Charge", "{"]
        for ind, charge in enumerate(self.m_charges):
            text_arr.append("{}".format(charge["angle"]))
            if charge["distance"] > 0:
                text_arr.append(":")
                text_arr.append("{}pt".format(int(charge["distance"])))
            text_arr.append("=")
            #text_arr.append(charge["charge"])
            if charge["charge"] in ["\\:", "\\|", "\\.", "\\"]:
                text_arr.append(charge["charge"])
            else:
                reverse_text, bad_trans = reverse_transcription(charge["charge"])
                if bad_trans:
                    text_arr.append(charge["charge"])
                else:
                    text_arr.append(reverse_text)
            if ind < len(self.m_charges) - 1:
                text_arr += [","]
        text_arr.append("}")
        text_arr.append("{")
        baseText, bad_trans = reverse_transcription(self.m_text)
        if not bad_trans:
            text_arr.append(baseText)  #fixed 2022.06.15
        else:
            text_arr.append(self.m_text)
        text_arr.append("}")
        return text_arr


class Bond(object):
    index = 0
    __default__ = {"m_angle": 0, "m_length": 1, "m_start": 0, "m_end": 0}

    def __init__(self, b_type="-"):
        self.name = "Bond_{}".format(Bond.index)
        Bond.index += 1
        b_type = b_type.replace("_", "").replace("^", "")

        self.m_type = b_type
        self.m_angle = None
        self.m_length = None
        self.m_start = None  #Bond.__default__["m_start"]
        self.m_end = None  #Bond.__default__["m_end"]
        self.m_extra_info = None
        self.super_info = None
        self.branch_info = [] # 存储键在环上相连的位置
        self.ring_branch_info = [] # 存储换上相连的键,即该bond的beginatom与branch相连

        self.begin_atom = None
        self.end_atom = None

        self.ring_ids = {}

        self.is_assigned = set()

    # def __setattr__(self, key, value):
    #     super(Bond, self).__setattr__(key, value)
    #     if value is not None and key in ["m_angle", "m_length"]:
    #         if "is_assigned" not in self.__dict__:
    #             self.__dict__["is_assigned"] = set()
    #         self.is_assigned.add(key)

    def __repr__(self):
        out_str = "<Bond"
        out_str += " name={}".format(self.name)
        out_str += " type={}".format(self.m_type)
        out_str += " angle={}".format(self.m_angle)
        out_str += " length={}".format(self.m_length)
        out_str += " start={}".format(self.m_start)
        out_str += " end={}".format(self.m_end)
        out_str += " begin_atom={}".format(self.begin_atom.name if self.begin_atom else "none")
        out_str += " end_atom={}".format(self.end_atom.name if self.end_atom else "none")
        out_str += ">"
        return out_str

    def parse_attr_from_str(self, attr_str):
        spts = re.split(",", attr_str[1:-1])
        if len(spts) > 0:
            angle_str = spts[0]
            angle_str = angle_str.replace(" ", "")
            if angle_str == "":
                pass
            elif angle_str.startswith("::"):  #relative angle
                self.m_angle = angle_str
            elif angle_str.startswith(":"):  #
                self.m_angle = int(float(angle_str[1:]))
                self.is_assigned.add("m_angle")
            else:
                angle_code = float(angle_str.replace("O", "0"))
                angle = 45 * angle_code
                self.m_angle = angle % 360
                self.is_assigned.add("m_angle")
        if len(spts) > 1:
            length_str = spts[1]
            if length_str == "":
                pass
            else:
                self.m_length = float(length_str)
                self.is_assigned.add("m_length")
        if len(spts) > 2:
            start_str = spts[2]
            if start_str == "":
                pass
            else:
                self.m_start = int(start_str)
        if len(spts) > 3:
            end_str = spts[3]
            if end_str == "":
                pass
            else:
                self.m_end = int(end_str)
        if len(spts) > 4:
            extra_info = spts[4]
            self.m_extra_info = extra_info
            if extra_info.replace(" ", "") == "draw=none":
                self.m_type = "-:"
            elif extra_info.replace(" ", "") == "mcfwavy":
                self.m_type = "~/"
            # else:
            #     raise NotImplementedError("not support extra_info of bond, {}".format(extra_info))

        if len(spts) > 5:
            raise NotImplementedError("not support more than 5 attrs of bond")
        pass


class AttrStr(object):
    def __init__(self, input):
        self.attr_str = input.replace(" ", "")
        pass

    def __repr__(self):
        outStr = "<AttrStr "
        outStr += " str={}".format(self.attr_str)
        outStr += " >"
        return outStr


class DistantHook(object):
    # _bond_dict = {1: "-", 2: "=", 3: "~", 4: ">", 5: "<", 6: ">:", 7: "<:", 8: ">|", 9: "<|"}
    _bond_dict = {1: "-", 2: "=", 3: "~", 4: ">", 5: "<", 6: ">:", 7: "<:", 8: ">|", 9: "<|", 10: "-@", 11: "=@", 12: "~@", 13: ">@", 14: "<@", 15: ">:@", 16: "<:@", 17: ">|@", 18: "<|@", 19: "-#", 20: "=#", 21: "~#", 22: ">#", 23: "<#", 24: ">:#", 25: "<:#", 26: ">|#", 27: "<|#",}
    _bond_set = set(_bond_dict.values())

    def __init__(self, input):
        self.attr_str = input
        self.m_hookname = ""
        self.m_bondtype = "-"  # modified at 2022.01.14
        self.parse_attr_from_str(input)
        pass

    def parse_attr_from_str(self, attr_str):
        attr_str = attr_str.replace(" ", "")
        assert attr_str.startswith("?[") and attr_str.endswith("]")
        spts = attr_str[2:-1].split(",")
        if len(spts) > 0:
            self.m_hookname = spts[0]
        if len(spts) > 1:
            if spts[1] in DistantHook._bond_set:
                self.m_bondtype = spts[1]
            elif spts[1].startswith("{") and spts[1].endswith("}") and spts[1][1:-1] in DistantHook._bond_set:
                self.m_bondtype = spts[1][1:-1]
            else:
                bond_code = int(spts[1])
                self.m_bondtype = DistantHook._bond_dict[bond_code]

    def __repr__(self):
        outStr = "<DistantHook "
        outStr += " name={}".format(self.m_hookname)
        outStr += " bondtype={}".format(self.m_bondtype)
        outStr += " >"
        return outStr


class RingHead(object):
    index = 0

    def __init__(self, r_type=""):
        self.name = "RingHead_{}".format(RingHead.index)
        RingHead.index += 1
        self.m_type = r_type
        self.m_num_edges = 1
        self.m_angle = 0
        self.m_start_angle = 0
        self.m_end_angle = 0
        self.root_atom = None

    def __repr__(self):
        out_str = "<RingHead"
        out_str += " name={}".format(self.name)
        out_str += " type={}".format(self.m_type)
        out_str += " n_edges={}".format(self.m_num_edges)
        out_str += " start_angle={}".format(self.m_start_angle)
        out_str += " end_angle={}".format(self.m_end_angle)
        out_str += ">"
        return out_str

    def parse_attr_from_str(self, attr_str):
        # '[30, 150]6'
        attr_str = "".join(list(filter(lambda x: x != " ", attr_str)))
        left_bracket_pos = attr_str.find("[")
        right_bracket_pos = attr_str.find("]")
        if left_bracket_pos != -1 and right_bracket_pos != -1:
            angle_str = attr_str[left_bracket_pos + 1:right_bracket_pos]
            spts = angle_str.split(",")
            #pdb.set_trace()
            if len(spts) > 0:
                self.m_start_angle = int(spts[0])
            if len(spts) > 1:
                self.m_end_angle = int(spts[1])
            self.m_num_edges = int(attr_str[right_bracket_pos + 1])
        else:
            self.m_num_edges = int(attr_str)
        pass

def calc_pos(pos_arr:list, ignore=["Branch", "Atom"]):
    pos = 0
    for _type in pos_arr[:-1]:
        if _type in ignore:
            continue
        pos += 1
    return pos


def calc_is_ring(parentInfo_arr:list):
    for parentInfo in reversed(parentInfo_arr):
        if parentInfo["is_ring"] is True:
            return parentInfo
        is_first = True
        for _type in reversed(parentInfo["pos_arr"][:-1]):
            if _type == "Bond" or _type == "Branch":
                is_first = False
                break
        if is_first is True:
            continue
        else:
            return None
    return None

def find_parent_base_angle(parentInfo_arr:list, cur_base_angle=None):
    base_angle = cur_base_angle
    if base_angle is None:
        base_angle = "::0"
    elif type(base_angle) is str:
        pass
    else:
        return base_angle
    # is_relative = False
    # if type(base_angle) is str:
    #     base_angle = float(base_angle[2:])
    #     is_relative = True
    for parentInfo in reversed(parentInfo_arr):
        if parentInfo["is_ring"] is True:
            return base_angle
        is_first = True
        pos = calc_pos(parentInfo["pos_arr"])
        if pos > 0:
            return base_angle
        else:
            _base_angle = parentInfo["m_base_angle"]
            if _base_angle is None:
                pass
            elif type(_base_angle) is str:
                cur_num = float(base_angle[2:])
                _num = float(_base_angle[2:])
                base_angle = "::{}".format(cur_num+_num)
            else:
                cur_num = float(base_angle[2:])
                base_angle = cur_num + _base_angle
                break
    return base_angle

    


class Branch(object):
    index = 0

    def __init__(self, atom):
        self.name = "Branch_{}".format(Branch.index)
        Branch.index += 1
        self.root_atom = atom
        self.m_base_angle = None
        self.m_base_length = None
        self.m_default_angle = None  # if not assign base_angle, use default_angle # calc by rule
        self.m_absolute_angle = None  #
        self.childs = []
        pass

    def parse_attr_from_str(self, attr_str):
        spts = re.split(",", attr_str[1:-1])
        if len(spts) > 0:
            angle_str = spts[0]
            if angle_str == "":
                pass
            elif angle_str.startswith("::"):  #relative angle
                self.m_base_angle = angle_str
                # pdb.set_trace()
            elif angle_str.startswith(":"):  #
                self.m_base_angle = int(float(angle_str[1:]))
            else:
                angle_code = int(angle_str)
                angle = 45 * angle_code
                self.m_base_angle = angle % 360
        if len(spts) > 1:
            length_str = spts[1]
            if length_str == "":
                pass
            else:
                self.m_base_length = float(length_str)

    # parentInfo: m_absolute_angle, is_ring
    def calc_angle(self, parentInfo:list=None, shared_info:dict=None):
        # is_in_ring
        parent_ring_info =  calc_is_ring(parentInfo)
        is_ring = parent_ring_info is not None
        lastBond = parentInfo[-1]["last_bond"]
        #pos = calc_pos(parentInfo[-1]["pos_arr"])

        # if self.m_base_angle is None:
        #     if calc_pos(parentInfo[-1]["pos_arr"]) == 0:
        #         self.m_base_angle = parentInfo[-1]["m_base_angle"]

        #base_angle = find_parent_base_angle(parentInfo, self.m_base_angle)
        base_angle = self.m_base_angle

        # parent
        if base_angle is None:  # use default
            if is_ring is True:
                ring_bond_angles = parent_ring_info["parent_ring_angles"]
                edge_num = len(ring_bond_angles)
                cur_bond_index = parent_ring_info["parent_ring_start_ind"]
                angleA = ring_bond_angles[(cur_bond_index) % edge_num]
                angleB = ring_bond_angles[(cur_bond_index+1) % edge_num]
                angleCenter = ((angleA+angleB-180) / 2.0) % 360
                angleDelta = (angleCenter-angleA) % 360
                if angleDelta < 180:
                    angleCenter = (angleCenter+180) % 360
                self.m_absolute_angle = angleCenter
            else:
                self.m_absolute_angle = parentInfo[-1]["m_absolute_angle"]
                
        elif type(base_angle) is not str:
            self.m_absolute_angle = base_angle
        else:
            relative_angle = int(float(base_angle[2:]))
            pre_angle = None
            # if lastBond is not None:
            #     pre_angle = lastBond.m_angle
            # elif calc_pos(parentInfo[-1]["pos_arr"]) == 0:
            #     pre_angle = parentInfo[-1]["m_absolute_angle"]
            # else:
            #     raise ValueError("File \"{}\", line {} : , no pre angle found for relative angles, {}".format(__file__, sys._getframe().f_lineno, self.m_base_angle))

            if calc_pos(parentInfo[-1]["pos_arr"]) == 0:
                pre_angle = parentInfo[-1]["m_absolute_angle"]
            elif lastBond is not None:
                pre_angle = lastBond.m_angle
            elif shared_info["is_root"] is False:
                pre_angle = 0
            else:
                raise ValueError("File \"{}\", line {} : , no pre angle found for relative angles, {}".format(__file__, sys._getframe().f_lineno, self.m_base_angle))

            # if calc_pos(parentInfo[-1]["pos_arr"]) == 0:
            #     if "m_absolute_angle" in parentInfo[-1]:
            #         pre_angle = parentInfo[-1]["m_absolute_angle"]
            #     else:
            #         raise ValueError("File \"{}\", line {} : , no pre angle found for relative angles, {}".format(__file__, sys._getframe().f_lineno, self.m_base_angle))
            # else:
            #     if lastBond is not None:
            #         pre_angle = lastBond.m_angle
            #     elif "m_absolute_angle" in parentInfo[-1]:
            #         pre_angle = parentInfo[-1]["m_absolute_angle"]
            #     else:
            #         raise ValueError("File \"{}\", line {} : , no pre angle found for relative angles, {}".format(__file__, sys._getframe().f_lineno, self.m_base_angle))
            self.m_absolute_angle = pre_angle + relative_angle
        return self.m_absolute_angle

    def build_child_graph(self, parentInfo:list=None, shared_info:dict=None):
        lastElement = self.root_atom
        # if lastElement is not None:
        #     if len(lastElement.in_bonds) == 1:
        #         lastBond = lastElement.in_bonds[0]  #todo maybe incorrect
        #     elif len(lastElement.in_bonds) == 0:
        #         lastBond = None
        #     else:
        #         # pdb.set_trace()
        #         # raise Warning("multiple in bond, we may can not decide")
        #         warnings.warn("multiple in bond, we may can not decide")
        #         # lastBond = lastElement.in_bonds[-1]  #we select newest
        #         lastBond = lastElement.in_bonds[0]  #we select oldest
        # else:
        #     lastBond = None
        lastBond = parentInfo[-1]["last_bond"]

        self.calc_angle(parentInfo, shared_info)

        shared_info["is_root"] = False

        base_parent_info = {}
        base_parent_info["m_absolute_angle"] = self.m_absolute_angle
        base_parent_info["m_base_angle"] = self.m_base_angle
        base_parent_info["is_ring"] = False
        base_parent_info["last_bond"] = parentInfo[-1]["last_bond"]

        type_list = []
        for i, child in enumerate(self.childs):
            if isinstance(child, Branch):
                type_list.append("Branch")
                assert isinstance(lastElement, Atom)
                next_parent_info = copy.deepcopy(base_parent_info)
                next_parent_info["pos_arr"] = copy.deepcopy(type_list)
                child.build_child_graph(parentInfo+[next_parent_info], shared_info)
            elif isinstance(child, Bond):
                type_list.append("Bond")
                assert isinstance(lastElement, Atom)
                # if child.name == "Bond_5":
                #     pdb.set_trace()
                #connect
                lastElement.out_bonds.append(child)
                child.begin_atom = lastElement
                #update angle
                if child.m_angle is not None:
                    pass
                elif self.m_absolute_angle is not None:
                    child.m_angle = self.m_absolute_angle
                else:
                    child.m_angle = Bond.__default__["m_angle"]
                if type(child.m_angle) is str:
                    relative_angle = int(child.m_angle[2:])
                    if lastBond is None:
                        child.m_angle = relative_angle
                    else:
                        child.m_angle = lastBond.m_angle + relative_angle
                # update length
                if child.m_length is not None:
                    pass
                elif self.m_base_length is not None:
                    child.m_length = self.m_base_length
                else:
                    child.m_length = Bond.__default__["m_length"]
                #update last element
                lastElement = child
                lastBond = child
                # if child.m_type not in ["-:", "~/"]:
                #     shared_info["is_bond_solved"] = True
                if child.m_length > 0:
                    shared_info["is_bond_solved"] = True
                base_parent_info["last_bond"] = child
            elif isinstance(child, Atom):
                # assert isinstance(lastElement, Bond) and lastElement is None, "Atom {} pre is not Bond or None, is {}".format(child, lastElement)
                #connect
                # pdb.set_trace()
                type_list.append("Atom")
                if lastElement is None:
                    self.root_atom = child
                elif isinstance(lastElement, Bond):
                    child.in_bonds.append(lastElement)
                    lastElement.end_atom = child
                elif isinstance(lastElement, Atom):
                    virtual_bond = Bond("")
                    virtual_bond.m_angle = 0
                    virtual_bond.m_length = 0
                    virtual_bond.m_type = "-:"
                    lastElement.out_bonds.append(virtual_bond)
                    virtual_bond.begin_atom = lastElement
                    virtual_bond.end_atom = child
                    child.in_bonds.append(virtual_bond)
                else:
                    raise TypeError("Atom {} pre is not Bond or None, is {}".format(child, lastElement))
                #update last element
                lastElement = child
                pass
            else:
                type_list.append("other")
                raise TypeError("[Build child graph] abnormal child type = {}".format(child.__class__.__name__))
        return self.root_atom

    def __repr__(self):
        outStr = "<Branch"
        outStr += " name={}".format(self.name)
        outStr += " root={}".format(self.root_atom)
        outStr += " baseAngle={}".format(self.m_base_angle)
        outStr += " absAngle={}".format(self.m_absolute_angle)
        outStr += " baseLength={}".format(self.m_base_length)
        outStr += ">"
        return outStr


class Ring(Branch):
    def __init__(self, atom, head=None):
        super(Ring, self).__init__(atom)
        self.ring_head = head
        if "is_assigned" not in self.__dict__:
            self.__dict__["is_assigned"] = set()
        pass

    # calc default angles
    def calc_angle(self, parentInfo:list=None, shared_info:dict=None):
        rootBond = parentInfo[-1]["last_bond"]
        edge_num = self.ring_head.m_num_edges
        inner_angle = 180 * (edge_num-2) / edge_num
        angle_step = float(360) / edge_num
        ring_bond_angles = [None for i in range(edge_num)]

        is_first = not shared_info["is_bond_solved"]
        parent_ring_info = calc_is_ring(parentInfo)
        is_ring = parent_ring_info is not None
        # pdb.set_trace()
        # is_first = True
        # for pos in parentInfo["pos"]:
        #     if pos > 0:
        #         is_first = False
        #         break
        # is_first = not parentInfo["is_bond_solved"]

        #calc last angle when absolute angle is 0°
        zero_last_angle = None
        if is_ring is True:
            parent_edge_num = len(parent_ring_info["parent_ring_angles"])
            parent_inner_angle = 180 * (parent_edge_num-2) / parent_edge_num
            zero_last_angle = (180 - parent_inner_angle + 180)%360
        else:
            if is_first is False:
                zero_last_angle = (inner_angle/2 + 180)%360
            else:
                zero_last_angle = 270

        # if self.name == "Branch_3":
        #     pdb.set_trace()

        # calc default angle
        self.m_default_angle = None
        if is_ring is True:
            #assert "parent_ring_angles" in parentInfo[-1] and "parent_ring_start_ind" in parentInfo[-1]
            if parent_ring_info["parent_ring_start_ind"] == -1:
                # self.m_default_angle = parentInfo["m_absolute_angle"]
                self.m_default_angle = parent_ring_info["m_absolute_angle"]
            else:
                parent_edge_num = len(parent_ring_info["parent_ring_angles"])
                out_index = (parent_ring_info["parent_ring_start_ind"] + 1) % parent_edge_num
                last_angle = (parent_ring_info["parent_ring_angles"][out_index] - 180) % 360
                self.m_default_angle = (last_angle - zero_last_angle)%360
        else:
            if rootBond is not None:
                last_angle = (rootBond.m_angle - 180 + inner_angle/2.0)
                self.m_default_angle = (last_angle - zero_last_angle)%360
            elif is_first is False:
                self.m_default_angle = 0
            else:
                self.m_default_angle = (270 - zero_last_angle)%360

        # apply base_angle_rule
        base_angle = self.m_base_angle
        base_angle = find_parent_base_angle(parentInfo, self.m_base_angle)
        # if base_angle is None:
        #     base_angle = parentInfo[-1]["m_base_angle"]
        if base_angle is None:  # use default
            # if parentInfo["pos"] == 0:
            #     self.m_absolute_angle = parentInfo["m_absolute_angle"]
            # else:
            #     self.m_absolute_angle = self.m_default_angle
            self.m_absolute_angle = self.m_default_angle
        elif type(base_angle) is not str:
            self.m_absolute_angle = base_angle
        else:
            relative_angle = int(float(base_angle[2:]))
            self.m_absolute_angle = self.m_default_angle + relative_angle
        
        last_angle = zero_last_angle + self.m_absolute_angle
        for i in range(edge_num):
            ring_bond_angles[i] = (last_angle - angle_step * (edge_num-1) + angle_step*i) % 360
        return ring_bond_angles

    # parentInfo is used for ring to decide angle of bonds
    def build_child_graph(self, parentInfo:list=None, shared_info:dict=None):
        lastElement = self.root_atom
        lastBond = parentInfo[-1]["last_bond"]
        # if lastElement is not None:
        #     if len(lastElement.in_bonds) == 1:
        #         lastBond = lastElement.in_bonds[0]  #todo maybe incorrect
        #     elif len(lastElement.in_bonds) == 0:
        #         lastBond = None
        #     else:
        #         raise Warning("multiple in bond, we may can not decide")
        #         # lastBond = lastElement.in_bonds[-1]  #we select newest
        #         lastBond = lastElement.in_bonds[0]  #we select oldest
        ring_bond_angles = self.calc_angle(parentInfo, shared_info)

        ignore_last_edge = False
        # if parentInfo and "ignore_last_edge" in parentInfo and parentInfo["ignore_last_edge"] is True:
        #     ignore_last_edge = True
        edge_num = self.ring_head.m_num_edges

        # next_parent_info = copy.deepcopy(parentInfo)
        # if parentInfo is None:
        #     next_parent_info = {"is_ring": 1}
        # else:
        #     next_parent_info = copy.deepcopy(parentInfo)
        #     next_parent_info["is_ring"] = 1
        base_parent_info = {}
        base_parent_info["m_absolute_angle"] = self.m_absolute_angle
        base_parent_info["m_base_angle"] = self.m_base_angle
        base_parent_info["parent_ring_angles"] = ring_bond_angles
        base_parent_info["parent_ring_start_ind"] = -1
        base_parent_info["is_ring"] = True
        base_parent_info["last_bond"] = parentInfo[-1]["last_bond"]
        # old_pos = copy.deepcopy(next_parent_info["pos"])

        shared_info["is_root"] = False

        cur_bond_index = 0
        type_list = []
        for i, child in enumerate(self.childs):
            if isinstance(child, Branch):
                type_list.append("Branch")
                assert isinstance(lastElement, Atom)
                next_parent_info = copy.deepcopy(base_parent_info)
                next_parent_info["parent_ring_start_ind"] = cur_bond_index - 1
                next_parent_info["pos_arr"] = copy.deepcopy(type_list)
                child.build_child_graph(parentInfo + [next_parent_info], shared_info)
                # if not isinstance(child, Ring):
                #     # if child.m_base_angle is not None: #may be wrong
                #     if child.m_base_angle is None:
                #         angleA = ring_bond_angles[(cur_bond_index-1) % edge_num]
                #         angleB = ring_bond_angles[cur_bond_index % edge_num]
                #         angleCenter = ((angleA+angleB-180) / 2.0) % 360
                #         angleDelta = (angleCenter-angleA) % 360
                #         if angleDelta < 180:
                #             angleCenter = (angleCenter+180) % 360
                #         child.m_base_angle = angleCenter
                #     child.build_child_graph(parentInfo)
                # else:
                #     child.build_child_graph(parentInfo)
            elif isinstance(child, Bond):
                type_list.append("Bond")
                assert isinstance(lastElement, Atom)
                #connect
                lastElement.out_bonds.append(child)
                child.begin_atom = lastElement
                #update angle
                # if child.m_angle is not None:
                #     pass
                # elif self.m_base_angle is not None:
                #     child.m_angle = ring_bond_angles[cur_bond_index]
                #     # if type(self.m_base_angle) is str:
                #     #     relative_base_angle = int(self.m_base_angle[2:])
                #     # child.m_angle += relative_base_angle
                #     # child.m_angle = self.m_base_angle + ring_bond_angles[cur_bond_index]
                # else:
                #     child.m_angle = ring_bond_angles[cur_bond_index]
                if ring_bond_angles is not None and cur_bond_index < len(ring_bond_angles):
                    child.m_angle = ring_bond_angles[cur_bond_index]
                else:
                    raise ValueError("File \"{}\", line {} : , can not find ring_bond_angles in ring, bond_id={}".format(__file__, sys._getframe().f_lineno, child.name))

                    # child.m_angle = Bond.__default__["m_angle"]
                # if type(child.m_angle) is str:
                #     relative_angle = int(child.m_angle[2:])
                #     if lastBond is None:
                #         child.m_angle = relative_angle
                #     else:
                #         child.m_angle = lastBond.m_angle + relative_angle
                # update length
                if child.m_length is not None:
                    pass
                elif self.m_base_length is not None:
                    child.m_length = self.m_base_length
                else:
                    child.m_length = Bond.__default__["m_length"]

                #update bond index
                if ignore_last_edge:
                    # pdb.set_trace()
                    # lastElement.out_bonds.remove(child)
                    if cur_bond_index >= edge_num - 1:  #last  bond
                        # pdb.set_trace()
                        lastElement.out_bonds.remove(child)
                        break
                    pass
                else:
                    if cur_bond_index >= edge_num - 1:  #last  bond
                        # pdb.set_trace()
                        child.end_atom = self.root_atom
                        self.root_atom.in_bonds.append(child)
                        break
                #update last element
                lastElement = child
                lastBond = child
                # increment
                cur_bond_index += 1
                if child.m_type not in ["-:", "~/"]:
                    shared_info["is_bond_solved"] = True
                base_parent_info["last_bond"] = child
            elif isinstance(child, Atom):
                # assert isinstance(lastElement, Bond) and lastElement is None, "Atom {} pre is not Bond or None, is {}".format(child, lastElement)
                #connect
                # pdb.set_trace()
                type_list.append("Atom")
                if lastElement is None:
                    self.root_atom = child
                elif isinstance(lastElement, Bond):
                    child.in_bonds.append(lastElement)
                    lastElement.end_atom = child
                elif isinstance(lastElement, Atom):
                    virtual_bond = Bond("")
                    virtual_bond.m_angle = 0
                    virtual_bond.m_length = 0
                    virtual_bond.m_type = "-:"
                    lastElement.out_bonds.append(virtual_bond)
                    virtual_bond.begin_atom = lastElement
                    virtual_bond.end_atom = child
                    child.in_bonds.append(virtual_bond)
                else:
                    raise TypeError("Atom {} pre is not Bond or Atom or None, is {}".format(child, lastElement))
                #update last element
                lastElement = child
                pass
            else:
                type_list.append("other")
                raise TypeError("[Build child graph] abnormal child type = {}".format(child.__class__.__name__))
        #deal for **
        if self.ring_head.m_type == "**":
            ref_angle = ((ring_bond_angles[0] + ring_bond_angles[-1] + 180) / 2.0) % 360
            delta_angle = (ref_angle - ring_bond_angles[0]) % 360
            if delta_angle > 180:
                ref_angle = (ref_angle+180) % 360

            virtual_bond = Bond("-:")
            virtual_bond.m_angle = ref_angle
            intersection_angle = (180 * (edge_num-2) / edge_num) / 2.0
            virtual_bond.m_length = lastBond.m_length * 0.5 / math.cos(intersection_angle * math.pi / 180.0)

            circle_atom = CircleAtom("")
            circle_atom.m_radius = virtual_bond.m_length * math.sin(intersection_angle * math.pi / 180.0) * 0.75

            self.root_atom.out_bonds.append(virtual_bond)
            virtual_bond.begin_atom = self.root_atom
            virtual_bond.end_atom = circle_atom
            circle_atom.in_bonds.append(virtual_bond)
        return self.root_atom

    def __repr__(self):
        outStr = "<ring.Branch"
        outStr += " root={}".format(self.root_atom)
        outStr += " head={}".format(self.ring_head)
        outStr += " baseAngle={}".format(self.m_base_angle)
        outStr += " absAngle={}".format(self.m_absolute_angle)
        outStr += " baseLength={}".format(self.m_base_length)
        outStr += ">"
        return outStr
