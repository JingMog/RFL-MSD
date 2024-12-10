import os, sys
import re
import pdb
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
import cv2

from utils import replace_chemfig, get_atom_group

bond_types = ["-", "=", "~", ">", "<", ">:", "<:", ">|", "<|", "-:", "=_", "=^", "~/"]
bond_types = sorted(bond_types, key=lambda x: -len(x))
virtual_types = ['{', '}', '(', ')', 'branch(', 'branch)']
match_virtual = {'}':'{', ')':'(', 'branch)':'branch('}


class Atom:
    index = 0

    def __init__(self, text=""):
        self.name = "Atom_{}".format(Atom.index)
        Atom.index += 1
        self.m_text = text
        self.pos_x = 0
        self.pos_y = 0
        self.ring_ids = {}
        self.conn_bonds = []

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

        self.begin_atom = None
        self.end_atom = None

        self.ring_ids = {}

        self.is_assigned = set()


def main(ssml: str):
    print(ssml)
    item_list = ssml.split()
    print(item_list)
    for item in item_list:
       print(item) 


def judge_str_item_type(item: str):
    bond_types = ["-", "=", "~", ">", "<", ">:", "<:", ">|", "<|", "-:", "=_", "=^", "~/"]
    virtual_types = ['{', '}', '(', ')', 'branch(', 'branch)']

    if '?' in item:
        begin_conn_pattern = re.compile(r'\?\[[a-zA-Z]\]')
        begin_result = begin_conn_pattern.findall(item)
        if len(begin_result) > 0:
            return 'reconn_begin'
        else:
            return 'reconn_end'

    for bond in bond_types:
        if bond in item:
            return 'bond_atom'
    
    atom_pattern = re.compile(r'[a-zA-Z]+|\\circle')
    atom_result = atom_pattern.findall(item)
    if len(atom_result) > 0:
        if atom_result[0] == item:
            return 'atom'
    
    if item == 'branch(':
        return 'branch_begin'
    if item == 'branch)':
        return 'branch_end'

    if item in virtual_types:
        return 'virtual'
    # for virtual in virtual_types:
    #     if virtual in item:
    #         return 'virtual'
    return 'atom'
    
    # print("item " + item + " is not matched  !!!!")

def attr_obtain(str):
    item_type = judge_str_item_type(str)
    if item_type == 'bond_atom':
        # bond_type bond_angle
        bond_type_pattern = re.compile(r'.*\[')
        bond_angle_pattern = re.compile(r'\d+')
        bond_type = bond_type_pattern.findall(str)[0][:-1]
        bond_angle = int(bond_angle_pattern.findall(str)[0])
        return bond_type, bond_angle
    elif item_type == 'atom':
        # atom name
        return str
    elif item_type == 'reconn_end':
        # reconn bond type
        bond_type_pattern = re.compile(r'\{(.+)\}')
        bond_type = bond_type_pattern.findall(str)[0]
        assert bond_type is not None, "bond_type is null."
        return bond_type
    else:
        print("Error in attr_obtain, type not defined.")
        sys.exit()

def build_graph(input_str, is_debug = False):
    
    chemfig_text, rep_dict, rep_text = replace_chemfig(input_str)
    graph_list = []
    for k, v in rep_dict.items():
        Graph = nx.Graph()
        # 遍历每一个分子, v
        item_list = v.split()[1:]
        item_list = get_atom_group(item_list)
        # print(item_list)
        virtual_stack = [] # 模拟括号堆栈,用于括号匹配

        cur_atom = None
        cur_bond = None

        reconn_begin_atom_dict = {} # 记录回连开始的原子
        branch_stack = [] # 分支回溯堆栈
        is_reconn = False # 回连标志,因为回连原子在回连标识之后,所以需要额外一个标识
        is_branch_end = False
        cur_reconn_tag = ''

        node_tag = 0
        branch_begin_tag = 0
        
        for ssml_item in item_list:
            ssml_item_type = judge_str_item_type(ssml_item)
            if is_debug:
                print("cur: ", ssml_item, ssml_item_type)
            if ssml_item_type == 'atom':
                Graph.add_node(node_tag, name=ssml_item)
            elif ssml_item_type == 'bond_atom':
                Graph.add_node(node_tag) # 创建新节点
                Graph.add_node(node_tag + 1)
                bond_type, bond_angle = attr_obtain(ssml_item)
                if is_branch_end:
                    Graph.add_edge(branch_begin_tag, node_tag + 1, bond_type = bond_type, angle = bond_angle)
                    is_branch_end = False
                else:
                    Graph.add_edge(node_tag, node_tag + 1, bond_type = bond_type, angle = bond_angle)
                node_tag += 1

            elif ssml_item_type == 'reconn_begin':
                is_reconn = True
                tag = ssml_item[2]
                cur_reconn_tag = tag
                reconn_begin_atom_dict[tag] = node_tag

            elif ssml_item_type == 'reconn_end':
                cur_reconn_tag = ssml_item[2]
                reconn_atom = reconn_begin_atom_dict[cur_reconn_tag] # 获取回连开始原子
                Graph.add_edge(reconn_atom, node_tag)
                
                del reconn_begin_atom_dict[cur_reconn_tag] # 从字典中删除处理完的回连记录
            elif ssml_item_type == 'branch_begin':
                branch_stack.append(node_tag) # begin branch, push stack
            elif ssml_item_type == 'branch_end':
                # branch_len = 1 + node_tag - branch_stack[-1]
                branch_begin_tag = branch_stack[-1] # get stack top
                branch_stack.pop() # end branch, pop stack
                is_branch_end = True

            elif ssml_item_type == 'virtual':
                if len(virtual_stack) == 0:
                    virtual_stack.append(ssml_item) # 入栈
                else:
                    cur_virtual = virtual_stack[-1] # 栈顶
                    if ssml_item not in match_virtual.keys(): # 左括号,入栈
                        virtual_stack.append(ssml_item)
                    else: # 右括号,匹配
                        virtual_stack.pop() # 括号匹配,出栈
        graph_list.append(Graph)
    return graph_list
