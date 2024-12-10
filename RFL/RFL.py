# -*- coding: utf-8 -*-
# @Time      : 2024/3/11s

from cond_render_main import process_cond_render
from cond_render import GetAllAtoms, chemfig_random_cond_parse
from image_render import rend
from text_render import rend_text
from RFL_utils import *
from utils import replace_chemfig
from graph_cmp import match_graph
from Tool_Formula.latex_norm.reverse_transcription import reverse_convert_chinese_to_uc
from datetime import datetime
from tqdm import tqdm
import copy
import cv2
import time
from collections import Counter

input_string = None

# 验证添加特殊token前后是否一致
old_cs_string = None
old_ring_branch_info = None
old_branch_info = None
old_cond_data = None

def chemfig2chemstem(label: str, show = False, add_extra_token = True, need_ring_num = False):
    '''
        chemfig(ssml)->chemstem,解析分子字符串,检测环结构并进行归并,生成分子骨干描述
        cs_label需要的辅助信息:分支连接的原子位置(将环视为原子时该信息丢失),
    '''
    out_unit = process_cond_render(label) # 解析chemfig label,区分分子和数学公式部分
    cs_string = []
    branch_info = []
    ring_branch_info = []
    cond_data = [] # 引导信息
    total_ring_num = 0
    # 遍历解析得到的单元
    for ind, element in enumerate(out_unit):
        if not isinstance(element, Atom):
            # 非分子部分直接并入out_units
            cs_string.append(element)
            branch_info.append(None)
            ring_branch_info.append(None)
            cond_data.append(-1) # 没有引导
        else: # 化学分子
            if need_ring_num:
                cur_cs_string, cur_branch_info, cur_ring_branch_info, cur_cond_data, cur_ring_num = get_chemstem(element, show = show, add_extra_token = add_extra_token, need_ring_num = need_ring_num)
            else:
                cur_cs_string, cur_branch_info, cur_ring_branch_info, cur_cond_data = get_chemstem(element, show = show, add_extra_token = add_extra_token, need_ring_num = need_ring_num)
                cur_ring_num = 0
            begin_index = len(cs_string) + 2
            # 补充\chemfig { * }
            cur_cond_data = [x + begin_index if x != -1 else x for x in cur_cond_data]
            cur_cs_string = ['\chemfig', '{'] + cur_cs_string + ['}']
            cur_branch_info = [None, None] + cur_branch_info + [None]
            cur_ring_branch_info = [None, None] + cur_ring_branch_info + [None]
            cur_cond_data = [-1, -1] + cur_cond_data + [-1]
            
            pre_const_len = 2 + len(cs_string)
            for index, unit in enumerate(cur_ring_branch_info):
                if unit is not None:
                    cur_ring_branch_info[index] = [x + pre_const_len for x in unit] # ring_branch_info全部加const

            cs_string += cur_cs_string
            branch_info += cur_branch_info
            ring_branch_info += cur_ring_branch_info
            cond_data += cur_cond_data
            total_ring_num += cur_ring_num
    if show:
        print("骨干字符串 cs_string: ", cs_string)
        # print("分支位置信息 branch_info: ", branch_info)
        # print("分支在环上的位置信息 ring_branch_info: ", ring_branch_info)
        # print("条件引导信息 cond_data: ", cond_data)
        print("所有环的数量: total_ring_num: ", total_ring_num)

    if need_ring_num:
        return cs_string, branch_info, ring_branch_info, cond_data, total_ring_num
    else:
        return cs_string, branch_info, ring_branch_info, cond_data

def chemstem2chemfig(cs_string, branch_info, ring_branch_info, cond_data, show=False, add_extra_token=True):
    '''骨干分子字符串+branch_info -> chemfig'''
    # cs_string = "\chemfig { H _ { 2 } O } + C = \chemfig { H _ { 2 } O } + \chemfig { O } + H + C".split(" ")
    in_units, in_branch_info_units, in_ring_branch_info_units, in_cond_data_units = process_cs_string2units(cs_string, branch_info, ring_branch_info, cond_data)
    out_units = [] # 得到输出的chemfig字符串
    pre_len = 0 # 记录cs_string的长度, 注意,不是chemfig的长度!
    for index, (chemfig_item, branch_info_item, ring_branch_info_item, cond_data_item) in enumerate(zip(in_units, in_branch_info_units, in_ring_branch_info_units, in_cond_data_units)):
        if chemfig_item[0] == "\chemfig": # 分子
            # 遍历ring_branch_info_item, 减去常量, 还原
            pre_const_len = 2 + pre_len
            pre_len += len(chemfig_item)
            for uid, unit in enumerate(ring_branch_info_item):
                # 修改ring_branch_info的index
                if unit is not None:
                    ring_branch_info_item[uid] = [x - pre_const_len for x in unit] # ring_branch_info全部减const
            for uid, unit in enumerate(cond_data_item):
                # 修改cond_data的index
                if unit != -1:
                    cond_data_item[uid] = unit - pre_const_len # cond_data全部减const
            
            cur_cs_string = chemfig_item[2: len(chemfig_item)-1] # 去掉chemfig标识
            cur_branch_info = branch_info_item[2: len(branch_info_item)-1]
            cur_ring_branch_info = ring_branch_info_item[2: len(ring_branch_info_item)-1]
            cur_cond_data = cond_data_item[2: len(cond_data_item)-1]
            chemfig_string = cs_string2chemfig(cur_cs_string, cur_branch_info, cur_ring_branch_info, cur_cond_data, show=show, add_extra_token=add_extra_token)
            chemfig_string = ['\chemfig', '{'] + chemfig_string + ['}']
            out_units += chemfig_string
        else:
            out_units.append(chemfig_item) # 数学公式
            pre_len += 1
    
    final_chemfig_string = " ".join(out_units)
    final_chemfig_string = reverse_convert_chinese_to_uc(final_chemfig_string) # 处理UC,转化为汉字

    return final_chemfig_string


def get_chemstem(root_atom: Atom, show=False, add_extra_token = True, need_ring_num = False):
    '''根据分子的图结构寻找最小环,逐步合并'''
    cs_string = []
    all_atoms = GetAllAtoms(root_atom)
    all_atoms = sorted(all_atoms, key=lambda x:x.pos_x * 10000 - x.pos_y) # 从左上角原子开始遍历
    if show:
        cv2.imwrite('rend_name.jpg', rend(all_atoms[0], scale=100, rend_name=1))
    
    # -------------------step1: DFS广搜环获取简单环路-----------------
    # ring_paths, ring_id = BFS_visit(start_atom = all_atoms[0], show=show)
    ring_paths, ring_id = DFS_visit(start_atom = all_atoms[0], show=show)
    ring_bond_dict = get_ring_bond_dict(ring_paths) # 获取环内所有的bond
    cur_ring_num = len(ring_paths)
    # -------------------step2: 解析各个环之间相邻关系 -------------------
    adj_ring_info, adj_ring_bond, adj_ring_num = parse_ring_adj_relation(ring_paths=ring_paths, ring_id=ring_id)
    if show:
        for ring_id, path in ring_paths.items():
            print("环 {}".format(ring_id))
            show_atom_list(path)
        # print("环的相邻原子信息:", adj_ring_info)
        # print("环的相邻边信息:", adj_ring_bond)
        print("每个环的相邻环数:", adj_ring_num)

    # -------------------step3: 获取每个环相连的分键 -------------------
    ring_adj_branch_info = get_ring_adj_branch_info(ring_paths = ring_paths)
    old_ring_adj_branch_info = get_ring_adj_branch_info(ring_paths = ring_paths, del_other_ring=False)


    
    # -------------------step4: 根据相邻关系与分支将环归纳为超原子与超边 -------------------
    ring_central_point = {rid: get_ring_central(ring_paths[rid]) for rid, path in ring_paths.items()} # 所有环的中心点
    adj_ring_num = dict(sorted(adj_ring_num.items(), key=lambda x: x[1]*100 + ring_central_point[x[0]][0] )) # 首先按照环相邻的数量,数量相同的情况下优先合并左侧的环
    ring_branch_info_record = []
    # TODO:目前版本暂时认为环最多相邻一个环, 这里如果环相邻过多可能还需要特殊处理,暂时先不管了
    cond_super_index = []
    multi_ring_cond_super_index = [] # 多环合并时,条件引导信息同样需要逆序
    merge_stack = [] # 记录合并的字符,以添加<ea>
    multi_ring_merge_stack = [] # 多环合并时需要先解码superatom,之后解superbond
    multi_ring_branch_info = {} # 多环合并边时,branch_info同样需要逆序解码, bond:[branch_info]

    double_ring_stack = [] # 两环只有一个公共原子的时候同样需要逆序合并
    double_ring_cond_stack = []
    double_ring_branch_info_stack = []

    super_id = 0 # 记录super原子的id
    unprocess_ring_id = list(adj_ring_num.keys())
    while unprocess_ring_id:
    # for ring_id_i, adj_num in adj_ring_num.items(): # 按照从左到右的顺序合并
        ring_id_i = unprocess_ring_id.pop(0)
        adj_num = adj_ring_num[ring_id_i]
        # 处理环i
        if adj_num == 0:
            # 如果无相邻环, 则归并为super_atom
            # super_atom = Atom(text="\Superatom" + str(super_id))
            atom_char = chr(ord('a') + super_id)
            super_atom = Atom(text="\Superatom" + atom_char)
            super_id += 1
            
            # 遍历adj_ring_info,判断是否出现两个环只有原子相邻的情况,如果有需要更新ring_paths
            # 如果两个环只有原子相邻的情况,也需要逆序合并
            old_atom = []
            atom_conn_ring_id = []
            for key, value in adj_ring_info.items():
                if ring_id_i in key and len(value) == 1:
                    old_atom.append(value[0])
                    if key[0] == ring_id_i:
                        atom_conn_ring_id.append(key[1])
                    else:
                        atom_conn_ring_id.append(key[0])
                    adj_ring_info[key] = [] # 替换完之后adj_ring_info不再有相邻原子
            
            cur_ring_path = sort_ring_atoms(ring_paths[ring_id_i])
            ring_adj_branch_info = get_ring_adj_branch_info(ring_paths = ring_paths)
            for bond in ring_adj_branch_info[ring_id_i]:
                if bond.m_type == '-:': # 苯环鲍林式中间的圆环当作环的一部分
                    continue
                
                if bond.begin_atom in cur_ring_path:
                    pre_begin_atom = bond.begin_atom
                    bond.begin_atom = super_atom
                    super_atom.out_bonds.append(bond) # 将原来的键添加到superatom中
                    bond.branch_info.append(get_atom_index_in_ring(cur_ring_path, pre_begin_atom)) # 补充键在环上的位置信息
                    # super_atom.all_bonds.append((bond, bond.m_angle, bond.begin_atom))
                elif bond.end_atom in cur_ring_path:
                    pre_begin_atom = bond.end_atom
                    bond.end_atom = super_atom
                    super_atom.in_bonds.append(bond)
                    bond.branch_info.append(get_atom_index_in_ring(cur_ring_path, pre_begin_atom)) # 补充键在环上的位置信息
                    # super_atom.all_bonds.append((bond, bond.m_angle, bond.end_atom))
                else:
                    raise ValueError
                
                # 只有在原始分子图上和环相连的分键才需要记录ring_branch_info, 合并中间过程不需要记录
                if bond in old_ring_adj_branch_info[ring_id_i]:
                    pre_begin_atom.ring_branch_info.append(bond) 

                if bond in pre_begin_atom.in_bonds:
                    pre_begin_atom.in_bonds.remove(bond) # 删除这个分支键与环的连接
                else:
                    pre_begin_atom.out_bonds.remove(bond) # 删除这个分支键与环的连接

            all_atoms.append(super_atom)

            # 更新ring_paths
            if old_atom and atom_conn_ring_id:
                for o_atom, r_id in zip(old_atom, atom_conn_ring_id):
                    ring_paths[r_id].remove(o_atom)
                    ring_paths[r_id].append(super_atom)
                double_ring_stack.append(cur_ring_path)
                double_ring_cond_stack.append(super_atom.m_text)
            else: # 如果不出现两环共点
                merge_stack.append(cur_ring_path) # 先加\Superatom
                cond_super_index.append(super_atom.m_text)

                if double_ring_stack:
                    merge_stack += double_ring_stack[::-1] # 逆序合并
                    double_ring_stack = [] # 清空
                if double_ring_cond_stack:
                    cond_super_index += double_ring_cond_stack[::-1]
                    double_ring_cond_stack = []
                if double_ring_branch_info_stack:
                    pass
                    double_ring_branch_info_stack = []

            # 如果存在逆序合并的边,就将其逆序加入
            if multi_ring_merge_stack:
                merge_stack += multi_ring_merge_stack[::-1] # 边的合并需要逆序
                multi_ring_merge_stack = [] # 清空
            if multi_ring_branch_info:
                for temp_bond, temp_branch_info in multi_ring_branch_info.items():
                    temp_bond.branch_info += temp_branch_info[::-1] # branch_info同样需要逆序
                multi_ring_branch_info = {} # 清空
            if multi_ring_cond_super_index:
                cond_super_index += multi_ring_cond_super_index[::-1] # cond_data同样需要逆序
                multi_ring_cond_super_index = [] # 清空
            
            if show:
                cv2.imwrite('rend_name.jpg', rend(all_atoms[-1], scale=100, rend_name=1))
                cv2.imwrite('cur_ring.jpg', rend(cur_ring_path[0], scale=100, rend_name=1))
        elif adj_num == 1:
            # 如果有相邻环,则归并为super_bond. 此时相邻关系发生改变,重新更新adj_ring_num
            # 注意需要判断两环之间公共边的数量

            cur_ring_path = sort_ring_atoms(ring_paths[ring_id_i])
            ring_i_adj_id = [] # 和ring_i相邻环的id
            for key, value in adj_ring_info.items():
                if len(value) > 0:
                    if key[0] == ring_id_i:
                        ring_i_adj_id.append(key[1])
                    elif key[1] == ring_id_i:
                        ring_i_adj_id.append(key[0])
            if show:
                print("环", ring_id_i, "相邻的环id", ring_i_adj_id)

            ring_id_j = ring_i_adj_id[0] # 归并的目标环
            common_bond = get_common_bond(adj_ring_bond, ring_id_i, ring_id_j)
            old_common_bond = common_bond
            if len(common_bond) == 1:
                common_bond = common_bond[0]
            else:
                common_bond = sorted(common_bond, key = lambda bond: get_bond_central(bond)[0])
                common_bond = common_bond[0]
            
            common_atom = [common_bond.begin_atom, common_bond.end_atom]
            # 将环i归并到环j上.公共边为common_bond
            
            # 处理环i相连的分支
            ring_adj_branch_info = get_ring_adj_branch_info(ring_paths = ring_paths)
            circle_bond = None
            circle_atom = None
            for bond in ring_adj_branch_info[ring_id_i]:
            
                if bond.m_type == '-:': # 苯环中间圆环,将其当作环的一部分
                    if bond.begin_atom in common_atom or bond.end_atom in common_atom:
                        if bond.begin_atom in cur_ring_path:
                            pre_begin_atom = bond.begin_atom
                            circle_atom = bond.end_atom
                            circle_bond = bond
                        else:
                            pre_begin_atom = bond.end_atom
                            circle_atom = bond.begin_atom
                            circle_bond = bond
                        if judge_point_in_polygon(cur_ring_path, circle_atom):
                            tmp_branch_info = get_atom_index_in_ring(cur_ring_path, pre_begin_atom)
                            bond.branch_info.append(tmp_branch_info)
                        else:
                            circle_atom = None
                            circle_bond = None
                    continue
                
                need_process = True
                # 判断相连的分支是为其他环的边,或者是否与其他环相连.若是则不需要处理
                for j in ring_i_adj_id:
                    if (bond in ring_bond_dict[j]) or (bond in ring_adj_branch_info[j]):
                        need_process = False
                        break
                
                if need_process:
                    # 将分支连到公共边上, 根据距离远近选择连接的原子
                    tgt_atom = select_tgt_atom_by_distance(cur_ring_path, common_bond, bond) # 这里是通过欧式距离选择合并的目标点,是否合理?
                    # tgt_atom = common_bond.begin_atom
                    if bond.begin_atom in cur_ring_path:
                        pre_begin_atom = bond.begin_atom
                        bond.begin_atom = tgt_atom
                        tgt_atom.out_bonds.append(bond)
                        pre_begin_atom.out_bonds.remove(bond)
                        # bond.branch_info.append(get_atom_index_in_ring(cur_ring_path, pre_begin_atom)) # 补充键在环上的位置信息
                        tmp_branch_info = get_atom_index_in_ring(cur_ring_path, pre_begin_atom) # 补充键在环上的位置信息
                        if bond in multi_ring_branch_info.keys():
                            multi_ring_branch_info[bond].append(tmp_branch_info)
                        else:
                            multi_ring_branch_info[bond] = [tmp_branch_info]

                    elif bond.end_atom in cur_ring_path:
                        pre_begin_atom = bond.end_atom
                        bond.end_atom = tgt_atom
                        tgt_atom.in_bonds.append(bond)
                        pre_begin_atom.in_bonds.remove(bond)
                        # bond.branch_info.append(get_atom_index_in_ring(cur_ring_path, pre_begin_atom)) # 补充键在环上的位置信息
                        tmp_branch_info = get_atom_index_in_ring(cur_ring_path, pre_begin_atom)
                        if bond in multi_ring_branch_info.keys():
                            multi_ring_branch_info[bond].append(tmp_branch_info)
                        else:
                            multi_ring_branch_info[bond] = [tmp_branch_info]
                        
                    else:
                        raise ValueError
                    
                    # 保存ring_branch_info信息
                    if bond in old_ring_adj_branch_info[ring_id_i]:
                        pre_begin_atom.ring_branch_info.append(bond)

            # 处理完环i相连的所有分支,并将其全部连接到公共边上之后
            # 将环i的与相邻环的所有边从图上删除(公共边仍然保留),并更新相邻信息
            # 重新创建两个公共边的原子,并将其与环相连
            common_bond.m_type = common_bond.m_type + '@' # 标识superbond

            old_common_endpoint = [bond.begin_atom for bond in old_common_bond] + [bond.end_atom for bond in old_common_bond]
            endpoint_counts = Counter(old_common_endpoint)
            old_common_middlepoint = [element for element, count in endpoint_counts.items() if count != 1] # 非端点
            old_common_endpoint = [element for element, count in endpoint_counts.items() if count == 1] # 端点
            old_endpoint1 = old_common_endpoint[0] # old端点1
            old_endpoint2 = old_common_endpoint[1] # old端点2, 需要将端点分开
            
            # 有多个common_bond时需要创建多个新Bond
            new_common_bond = []
            middle_point_dict = {atom: None for atom in old_common_middlepoint} # old_atom: new_atom
            endpoint_new_bond_dict = {old_endpoint1: None, old_endpoint2: None} # old endpoint: new_bond
            for old_bond in old_common_bond:
                bond_type = old_bond.m_type.replace('@', '') + '#'
                new_bond = Bond(bond_type)
                new_bond.m_angle = old_bond.m_angle
                new_bond.m_length = old_bond.m_length
                new_common_bond.append(new_bond)
                
                # 处理非端点
                if old_bond.begin_atom in middle_point_dict.keys():
                    if middle_point_dict[old_bond.begin_atom] is None:
                        new_middle_atom = Atom(text=old_bond.begin_atom.m_text)
                        middle_point_dict[old_bond.begin_atom] = new_middle_atom
                    else:
                        new_middle_atom = middle_point_dict[old_bond.begin_atom]
                    new_bond.begin_atom = new_middle_atom
                    new_middle_atom.out_bonds.append(new_bond)
                if old_bond.end_atom in middle_point_dict.keys():
                    if middle_point_dict[old_bond.end_atom] is None:
                        new_middle_atom = Atom(text=old_bond.end_atom.m_text)
                        middle_point_dict[old_bond.end_atom] = new_middle_atom
                    else:
                        new_middle_atom = middle_point_dict[old_bond.end_atom]
                    new_bond.end_atom = new_middle_atom
                    new_middle_atom.in_bonds.append(new_bond)
                
                # 维护endpoint_new_bond_dict
                if old_bond.begin_atom in old_common_endpoint:
                    endpoint_new_bond_dict[old_bond.begin_atom] = new_bond
                if old_bond.end_atom in old_common_endpoint:
                    endpoint_new_bond_dict[old_bond.end_atom] = new_bond

            tmp_insert_list = []
            tmp_remove_list = []
            for atom in cur_ring_path:
                # print(atom)
                # if atom == common_bond.begin_atom:
                if atom == old_endpoint1:
                    cur_new_bond = endpoint_new_bond_dict[old_endpoint1]
                    new_begin_atom = Atom(atom.m_text) # 新的begin_atom
                    cur_new_bond.begin_atom = new_begin_atom
                    new_begin_atom.out_bonds.append(cur_new_bond)
                    for bond in atom.in_bonds + atom.out_bonds:
                        # print(bond)
                        if (bond.begin_atom in cur_ring_path) and (bond.end_atom in cur_ring_path) and (bond not in old_common_bond):
                            if bond.begin_atom == atom:
                                bond.begin_atom = new_begin_atom
                                new_begin_atom.out_bonds.append(bond)
                            else:
                                bond.end_atom = new_begin_atom
                                new_begin_atom.in_bonds.append(bond)
                            
                            if bond in atom.in_bonds:
                                atom.in_bonds.remove(bond)
                            else:
                                atom.out_bonds.remove(bond)
                    tmp_insert_list.append(new_begin_atom)
                    tmp_remove_list.append(atom)
                # elif atom == common_bond.end_atom:
                elif atom == old_endpoint2:
                    cur_new_bond = endpoint_new_bond_dict[old_endpoint2]
                    new_end_atom = Atom(atom.m_text) # 新的end_atom
                    cur_new_bond.end_atom = new_end_atom
                    new_end_atom.in_bonds.append(cur_new_bond)
                    for bond in atom.in_bonds + atom.out_bonds:
                        if (bond.begin_atom in cur_ring_path) and (bond.end_atom in cur_ring_path) and (bond not in old_common_bond):
                            if bond.begin_atom == atom:
                                bond.begin_atom = new_end_atom
                                new_end_atom.out_bonds.append(bond)
                            else:
                                bond.end_atom = new_end_atom
                                new_end_atom.in_bonds.append(bond)
                            
                            if bond in atom.in_bonds:
                                atom.in_bonds.remove(bond)
                            else:
                                atom.out_bonds.remove(bond)
                    tmp_insert_list.append(new_end_atom)
                    tmp_remove_list.append(atom)
                # 创建新的键并将两个环分开
            # 更新cur_ring_path
            for remove_item, insert_item in zip(tmp_remove_list, tmp_insert_list):
                cur_ring_path.remove(remove_item)
                cur_ring_path.append(insert_item)
            for old_atom, new_atom in middle_point_dict.items():
                cur_ring_path.remove(old_atom)
                cur_ring_path.append(new_atom)

            if circle_bond:
                if circle_bond.begin_atom in common_atom or circle_bond.end_atom in common_atom:
                    # 如果circle键与common_bond相连, 则需要将其连到分割出来的环上
                    tgt_atom = get_index_atom_in_ring(cur_ring_path, circle_bond.branch_info.pop(0))
                    if circle_atom == circle_bond.begin_atom:
                        pre_begin_atom = circle_bond.end_atom
                        pre_begin_atom.in_bonds.remove(circle_bond)
                        circle_bond.end_atom = tgt_atom
                        tgt_atom.in_bonds.append(circle_bond)
                    else:
                        pre_begin_atom = circle_bond.begin_atom
                        pre_begin_atom.out_bonds.remove(circle_bond)
                        circle_bond.begin_atom = tgt_atom
                        tgt_atom.out_bonds.append(circle_bond)


            multi_ring_merge_stack.append(cur_ring_path)
            multi_ring_cond_super_index.append([common_bond.m_type, common_bond.m_angle, common_bond])
            # merge_stack.append(cur_ring_path)
            # 更新相邻信息
            if ring_id_i > ring_id_j:
                ring_id_i, ring_id_j = ring_id_j, ring_id_i
            adj_ring_info[(ring_id_i, ring_id_j)] = [] # 环i,j相邻的原子清空
            adj_ring_bond[(ring_id_i, ring_id_j)] = [] # 环i,j相邻的边清空
            adj_ring_num[ring_id_i] -= 1 # 环i相邻的环--
            adj_ring_num[ring_id_j] -= 1 # 环j相邻的环--
            unprocess_ring_id = sorted(unprocess_ring_id, key=lambda x:adj_ring_num[x]*100 + ring_central_point[x][0] ) # 如果相邻数量一致的情况下,先合并左边的原子
            # sorted(adj_ring_num.items(), key=lambda x: x[1]*100 + ring_central_point[x[0]][0] )
            if show:
                cv2.imwrite('rend_name.jpg', rend(all_atoms[-1], scale=100, rend_name=1))
                cv2.imwrite('cur_ring.jpg', rend(cur_ring_path[0], scale=100, rend_name=1))
            rend(all_atoms[-1], scale=100, rend_name=1)
            rend(cur_ring_path[0], scale=100, rend_name=1)
        else:
            raise ValueError("adj_num大于1")
    # --------------------------step5: 根据分子骨干生成cs_string以及branch_info -------------------
    # 维护一个bond的字典,将每个Bond对应的index保存下来
    bond_dict = [] # {bond:index,}

    # ring_branch_info = [] # 在渲染的同时获取环上位置, 希望实现ring_branch_info和branch_info的转换。
    cs_string, branch_info, ring_branch_info, bond_dict = rend_text_with_branch_info(all_atoms[-1], bond_dict=bond_dict) # 同时获取分支在环上所连位置
    cs_string.append('<ea>')
    branch_info.append(None)
    ring_branch_info.append(None)
    # need_renumber = False # 当出现Superatom的嵌套时需要进行重编号
    # nest_super_atom_id = [] # 记录嵌套的Superatom id
    cond_data = [-1] * len(cs_string) # 条件引导,主干分支上没有condition引导,需要给单独的环添加条件引导
    
    if show:
        print(cond_super_index)
    while merge_stack:
        cur_ring = merge_stack.pop(0)
        if show:
            cv2.imwrite('cur_ring.jpg', rend(cur_ring[0], scale=100, rend_name=1))
        # 获取cur_ring的字符串
        cur_cs_string, cur_branch_info, cur_ring_branch_info, cur_bond_dict = rend_text_with_branch_info(cur_ring[0], bond_dict=[])
        cur_super_atom = cond_super_index.pop(0) # 当前Super标识
        
        # 获取cond_data
        super_index = None
        if isinstance(cur_super_atom, str): # superatom
            # super_id = cur_super_atom[10:] # superid
            for id, unit in enumerate(cs_string):
                # if unit == '\Superatom' and cs_string[id+1] == super_id:
                if unit == cur_super_atom:
                    super_index = id
                    break
        else: # superbond
            bond_type = cur_super_atom[0]
            bond_angle = cur_super_atom[1]
            common_bond = cur_super_atom[2] # 直接获取common_bond
            for bond_item in bond_dict: # 直接根据bond_dict获得super_index作为cond info
                if bond_item[0] == common_bond:
                    super_index = bond_item[1]
                    break
            # for id, unit in enumerate(cs_string):
            #     if unit.startswith(bond_type) and unit.endswith(']'):
            #         left_index = unit.index('[')
            #         right_index = unit.index(']')
            #         unit_angle = int(unit[left_index+2:right_index])
            #         if abs(unit_angle - bond_angle) < 20 or (abs(unit_angle - bond_angle) > 160 and abs(unit_angle - bond_angle) < 200):
            #             super_index = id
            #             break
            #     elif unit.startswith('?[') and unit.endswith(']') and ',' in unit: # reconn
            #         # ?[a,{=@}]
            #         index1 = unit.index('{')
            #         index2 = unit.index('}')
            #         if unit[index1+1:index2] == bond_type:
            #             super_index = id
            #             break
        assert super_index is not None, "Super index为空"
        # cur_cond_data = [super_index] * len(cur_cs_string)
        cur_cond_data = [-1] * len(cur_cs_string)


        cur_cs_string.append('<ea>')
        cur_branch_info.append(None)
        cur_cond_data.append(-1)
        cur_ring_branch_info.append(None)

        pre_length = len(cs_string)
        cs_string = cs_string + cur_cs_string
        branch_info = branch_info + cur_branch_info
        # cond_data[-1] = super_index
        cur_cond_data[0] = super_index
        cond_data = cond_data + cur_cond_data
        ring_branch_info = ring_branch_info + cur_ring_branch_info
        # 合并bond_dict
        for tmp_index, tmp_value in enumerate(cur_bond_dict):
            tmp_value[1] = pre_length + tmp_value[1]
            bond_dict.append([tmp_value[0], tmp_value[1]]) # 加入到bond_dict中
            # bond_dict[tmp_key] = cur_bond_dict[tmp_key] 
        
    # 根据bond_dict将cur_ring_branch_info转换为字符串对应的索引
    for info in ring_branch_info:
        if info is not None and len(info) > 0:
            for i in range(len(info)):
                info[i] = [tmp_value[1] for tmp_value in bond_dict if tmp_value[0] == info[i]][0]
                
                # for tmp_value in bond_dict:
                #     if tmp_value[0] == info[i]:
                #         info[i] = tmp_value[1] # bond->index
                #         break
    
    
    # 粗粒度匹配两种branch_info
    branch_info_num = 0
    ring_branch_info_num = 0
    for item in branch_info:
        if item is not None:
            branch_info_num += len(item)
    for item in ring_branch_info:
        if item is not None:
            ring_branch_info_num += len(item)
    # assert branch_info_num == ring_branch_info_num, "两种branch_info发生不匹配!"
    
    # --------------------------step6: 后处理 -------------------
    # 在cs_string中环上添加n个额外的token, '\connbranch', 表示这里有ring_branch_info需要预测
    if add_extra_token:
        global old_cs_string
        global old_branch_info
        global old_ring_branch_info
        global old_cond_data
        old_cs_string = copy.deepcopy(cs_string)
        old_branch_info = copy.deepcopy(branch_info)
        old_ring_branch_info = copy.deepcopy(ring_branch_info)
        old_cond_data = copy.deepcopy(cond_data)

        add_token_index_list = []
        ring_value_list = []
        cond_value_list = []
        for tmp_index, tmp_unit in enumerate(ring_branch_info):
            if tmp_unit is not None:
                add_token_index_list += [tmp_index] * len(tmp_unit)
                ring_value_list += tmp_unit
        
        for tmp_index, tmp_unit in enumerate(cond_data):
            if tmp_unit != -1 and tmp_unit not in cond_value_list:
                cond_value_list.append(tmp_unit)
        
        # 逆序添加token
        for add_index in add_token_index_list[::-1]:
            # cs_string = cs_string[:add_index+1] + ['\connbranch'] + cs_string[add_index+1:] 
            cs_string.insert(add_index+1, '\connbranch') # 在cs_string中添加\connbranch token
            
            ring_branch_info.insert(add_index+1, None)
            if add_index < max(ring_value_list):
                # print("需要修改ring_branch_info的值")
                # 如果add_index插入到ring_value之前,则将所有小于add_index的value ++
                for tmp_index, tmp_value in enumerate(ring_branch_info):
                    if tmp_value:
                        for m_index, m_value in enumerate(tmp_value):
                            if add_index < m_value:
                                ring_branch_info[tmp_index][m_index] += 1

            insert_value = cond_data[add_index]
            cond_data.insert(add_index+1, insert_value) # 维护cond_data
            # cond_data的value也需要修改
            if add_index < max(cond_value_list):
                for tmp_index, tmp_value in enumerate(cond_data):
                    if add_index < tmp_value:
                        cond_data[tmp_index] += 1

            branch_info.insert(add_index+1, None) # branch_info不需要额外的处理,因为保存的是相对位置
    # 去掉Superatom的后缀
    for index, unit in enumerate(cs_string):
        if '\Superatom' in unit:
            cs_string[index] = '\Superatom'

    # 验证
    assert len(cs_string) == len(branch_info), "cs_string长度必须和branch_info长度相同"
    assert len(cs_string) == len(cond_data), "cs_string长度必须和引导信息长度相同"
    assert len(cs_string) == len(ring_branch_info), "cs_string长度必须和ring_branch_info长度相同"
    for tmp_index, tmp_value in enumerate(cond_data):
        if tmp_value != -1:
            if not('\Superatom' in cs_string[tmp_value] or '@' in cs_string[tmp_value]):
                raise ValueError("cond_data信息有误")
    

    return cs_string, branch_info, ring_branch_info, cond_data


def cs_string2chemfig(cs_string, branch_info, ring_branch_info, cond_data, show=False, add_extra_token=True):
    '''根据cs_string以及branch_info还原回chemfig'''
    # --------------------------step0: 预处理删去\conntoken -------------------
    if add_extra_token:
        del_token_index_list = []
        ring_value_list = []
        cond_value_list = []
        for tmp_index, tmp_unit in enumerate(cs_string):
            if tmp_unit == '\connbranch':
                del_token_index_list.append(tmp_index)
        for tmp_index, tmp_unit in enumerate(ring_branch_info):
            if tmp_unit is not None:
                ring_value_list += tmp_unit
        for tmp_index, tmp_unit in enumerate(cond_data):
            if tmp_unit != -1:
                cond_value_list.append(tmp_unit)

        # 逆序删除
        for del_index in del_token_index_list[::-1]:
            cs_string.pop(del_index) # 维护cs_string
            
            ring_branch_info.pop(del_index) # ring_branch_info中删除connbranch对应的token
            cond_data.pop(del_index) # cond_data中删除connbranch对应的token
            if del_index < max(ring_value_list):
                # print("需要修改ring_branch_info的值")
                for tmp_index, tmp_value in enumerate(ring_branch_info):
                    if tmp_value:
                        for m_index, m_value in enumerate(tmp_value):
                            if del_index < m_value:
                                ring_branch_info[tmp_index][m_index] -= 1
            if del_index < max(cond_value_list):
                # print("需要修改cond_data的值")
                for tmp_index, tmp_value in enumerate(cond_data):
                    if del_index < tmp_value:
                        cond_data[tmp_index] -= 1

            branch_info.pop(del_index) # 因为branch_info保存的是相对位置,所以不需要修改

        if show:
            print("cs_string转换:", old_cs_string == cs_string)
            print("ring_branch_info转换:", old_ring_branch_info == ring_branch_info)
            print("cond_data转换:", old_cond_data == cond_data)

    # --------------------------step0: 获取\super和<ea>索引 -------------------
    merge_token_stack = []
    merge_token = ('\Superatom', '@')
    ea_token_stack = []
    
    for i in range(len(cs_string)):
        for t in merge_token:
            if t in cs_string[i]:
                merge_token_stack.append(i)
        if cs_string[i] == '<ea>':
            ea_token_stack.append(i)
    # if show:
    #     print("合并token的id:", merge_token_stack) # TODO:使用@标识会导致多一个token
    #     print("ea token的id:", ea_token_stack)

    # 
    # 维护一个dict,在渲染成图时保存连接目标对应的Bond实体,
    # 后续遇到有ring_branch_info的位置时根据bond_dict直接将对应的Bond实体保存到Atom的ring_branch_info中
    # 就可以直接获得tgt_atom

    # --------------------------step1: 将cs_string渲染成骨干图 -------------------
    conn_dict = {} # ring_branch index对应的token
    bond_dict = [] # 保存所有的bond在原始cs_string中对应的位置
    atom_dict = [] # 只保存Superatom
    for item in ring_branch_info:
        if item is not None:
            for t in item:
                conn_dict[t] = cs_string[t]
    
    stem_str = cs_string[: ea_token_stack[0]] # 骨干字符串
    stem_branch_info = branch_info[: ea_token_stack[0]] # 骨干branch_info
    stem_ring_branch_info = ring_branch_info[: ea_token_stack[0]] # 骨干ring_branch_info
    stem_cond_data = cond_data[: ea_token_stack[0]] # 骨干cond_data

    stem_str = " ".join(stem_str)
    root_atom = chemfig_random_cond_parse(stem_str, bond_dict=bond_dict, atom_dict=atom_dict, preprocess=False) # 在这里将branch_info添加到Bond中
    # 根据bond_dict和elements将ring_branch_info赋值到bond之前
    if stem_ring_branch_info:
        for i, item in enumerate(stem_ring_branch_info):
            if item is not None:
                tgt_bond = [k for k, v in bond_dict.items() if v == i][0]
                tgt_bond.begin_atom.ring_branch_info = item # 将ring_branch_info添加到bond的begin atom中
    
    stem_all_atoms = GetAllAtoms(root_atom)
    if show:
        cv2.imwrite('rend_name.jpg', rend(root_atom, scale=100, rend_name=1))
    # super_id = 1 # 使用cond_data还原就不再需要记录super_id了
    
    # --------------------------step2: 依次将归并的环加上 -------------------
    intermediate_branch = {} # 记录多环合并时中间没有保存ring_branch_info的分支, bond: begin_atom
    while merge_token_stack:
        # 拿出第一个环
        cur_super_index = merge_token_stack.pop(0) # 第一个superatom
        cur_ea_index = ea_token_stack.pop(0) # 第一个ea
        cur_ea_index2 = ea_token_stack[0] # 第二个ea
        cur_ring = cs_string[cur_ea_index + 1: cur_ea_index2]
        cur_branch_info = branch_info[cur_ea_index + 1: cur_ea_index2]
        cur_ring_branch_info = ring_branch_info[cur_ea_index + 1: cur_ea_index2]
        cur_cond_data = cond_data[cur_ea_index + 1: cur_ea_index2] # 每个环对应一个super_token

        # 在解析成图结构的时候加上branch_info, ring_branch_info
        cur_bond_dict = []
        cur_atom_dict = []
        ring_atom = chemfig_random_cond_parse(" ".join(cur_ring), bond_dict=cur_bond_dict, atom_dict=cur_atom_dict)
        if show:
            cv2.imwrite('cur_ring.jpg', rend(ring_atom, scale=100, rend_name=1))
        
        for index, value in enumerate(cur_bond_dict): # 用cur_bond_dict更新bond_dict
            bond_dict.append([value[0], value[1] + cur_ea_index + 1 ])
        for index, value in enumerate(cur_atom_dict): # 更新atom_dict
            atom_dict.append([value[0], value[1] + cur_ea_index + 1])

        if cur_ring_branch_info:
            for i, item in enumerate(cur_ring_branch_info):
                while item:
                    info = item.pop(0)
                    pre_index = i + cur_ea_index + 1 # 原始在cs_string中的index
                    tgt_bond = [item[0] for item in bond_dict if item[1] == pre_index][0]
                    tgt_bond.begin_atom.ring_branch_info.append(info) # 将ring_branch_info添加到bond的begin atom中
        ring_atom_list = GetAllAtoms(ring_atom)
        
        # 展开Superatom, 通过cond_data获取Superatom
        if '\Superatom' in cs_string[cur_cond_data[0]]: # 根据cond_data进行还原而不是按照顺序还原
            # 获取super_atom
            super_atom = None
            for item in atom_dict:
                if item[1] == cur_cond_data[0]:
                    super_atom = item[0]
                    break
            assert super_atom is not None, "未从cs_string中找到Superatom."
            
            # 使用ring_branch_info获得目标信息, ring_branch_info的顺序是无关的, 可以随意打乱
            tgt_atom = None
            # TODO:这里如何区分普通的Superatom和两环只有一个原子相邻的情况?
            # if super_atom in 

            intermediate_branch.update({bond: super_atom for bond in super_atom.in_bonds + super_atom.out_bonds})
            for atom in ring_atom_list:
                if atom.ring_branch_info is not None and len(atom.ring_branch_info) > 0:
                    # 找到目标键
                    while atom.ring_branch_info:
                        info = atom.ring_branch_info.pop(0)
                        tgt_bond = [item[0] for item in bond_dict if item[1] == info][0] # 根据ring_branch_info获取目标分支
                        
                        # 修改bond所连目标
                        if tgt_bond.begin_atom == super_atom:
                            tgt_bond.begin_atom = atom
                            atom.out_bonds.append(tgt_bond)
                        elif tgt_bond.end_atom == super_atom:
                            tgt_bond.end_atom = atom
                            atom.in_bonds.append(tgt_bond)
                        elif tgt_bond in intermediate_branch.keys():
                            if tgt_bond.begin_atom == intermediate_branch[tgt_bond]:
                                tgt_bond.begin_atom = atom
                                atom.out_bonds.append(tgt_bond)
                            elif tgt_bond.end_atom == intermediate_branch[tgt_bond]:
                                tgt_bond.end_atom = atom
                                atom.in_bonds.append(tgt_bond)
                        # else:
                        #     raise ValueError
                        
                        # 删除superatom所连的bond
                        if tgt_bond in super_atom.in_bonds:
                            super_atom.in_bonds.remove(tgt_bond)
                        elif tgt_bond in super_atom.out_bonds:
                            super_atom.out_bonds.remove(tgt_bond)
                        elif tgt_bond in intermediate_branch[tgt_bond].in_bonds:
                            intermediate_branch[tgt_bond].in_bonds.remove(tgt_bond)
                        elif tgt_bond in intermediate_branch[tgt_bond].out_bonds:
                            intermediate_branch[tgt_bond].out_bonds.remove(tgt_bond)

                        intermediate_branch.pop(tgt_bond)
            for bond in intermediate_branch.keys():
                # 将合并中间过程的这些bond全部添加到ring_atom_list[0]上
                cur_atom = ring_atom_list[0]
                tgt_bond = bond
                if tgt_bond.begin_atom == super_atom:
                    tgt_bond.begin_atom = cur_atom
                    cur_atom.out_bonds.append(tgt_bond)
                elif tgt_bond.end_atom == super_atom:
                    tgt_bond.end_atom = cur_atom
                    cur_atom.in_bonds.append(tgt_bond)
                
                intermediate_branch[tgt_bond] = cur_atom
                if tgt_bond in super_atom.in_bonds:
                    super_atom.in_bonds.remove(tgt_bond)
                elif tgt_bond in super_atom.out_bonds:
                    super_atom.out_bonds.remove(tgt_bond)



            # 环有相连分支,则展开之后Superatom需要删除掉
            # 从stem_all_atoms中删除super_atom
            if super_atom in stem_all_atoms: # 这里判断是为了尽可能还原(即使出错)
                stem_all_atoms.remove(super_atom)
            if 'Super' in root_atom.m_text:
                root_atom = ring_atom_list[0] # 如果当前root为SuperAtom,将其替换为环上的分子
                # root_atom = tgt_atom # 如果当前root_atom是super_atom,则该原子已经被删除,需要更新root_atom

            # 更新stem_all_atoms
            stem_all_atoms = GetAllAtoms(root_atom)
            if show:
                cv2.imwrite('rend_name.jpg', rend(stem_all_atoms[0], scale=100, rend_name=1))
        # 展开superbond
        elif '@' in cs_string[cur_cond_data[0]]:
            # common_bond支持多公共边, 2024.06.29
            common_bond = search_super_bond_in_ring(ring_atom_list) # 将cur_ring加到stem_all_atom上,公共边为common_bond
            common_bond_in_stem, ring_stem_atom_dict, stem_ring_bond_dict = search_super_bond_in_stem(stem_all_atoms, common_bond) # 在cur_ring中找到common_bond对应的bond, 同时返回atom, bond之间的对应关系
            another_bond_in_ring1, another_bond_in_ring2 = get_adj_bond(common_bond)
            
            circle_atom = None
            circle_bond = None
            all_ring_common_atom = set(list([atom for bond in common_bond for atom in [bond.begin_atom, bond.end_atom]])) # 公共边上的所有原子
            # 如果circle键连到common_bond上需要特殊处理
            for atom in ring_atom_list:
                if atom.m_text == '\circle':
                    for bond in atom.in_bonds + atom.out_bonds:
                        if bond.begin_atom == atom and bond.end_atom in all_ring_common_atom:
                            circle_bond = bond
                            circle_atom = atom
                            tmp_branch_info = get_atom_index_in_ring(ring_atom_list, bond.end_atom) # 补充键在环上的位置信息
                            circle_bond.branch_info.append(tmp_branch_info)
                            break
                        elif bond.end_atom == atom and bond.begin_atom in all_ring_common_atom:
                            circle_bond = bond
                            circle_atom = atom
                            tmp_branch_info = get_atom_index_in_ring(ring_atom_list, bond.begin_atom) # 补充键在环上的位置信息
                            circle_bond.branch_info.append(tmp_branch_info)
                            break
                    break
            
            ring_atom_counts = Counter([atom for bond in common_bond for atom in [bond.begin_atom, bond.end_atom]])
            ring_end_point = [element for element, count in ring_atom_counts.items() if count == 1] # 环上端点
            for bond in [another_bond_in_ring1, another_bond_in_ring2]:
                if bond.begin_atom in ring_end_point:
                    # 两个公共边合并前需要判断其方向是否相同(common_bond, common_bond_in_stem)
                    pre_ring_atom = bond.begin_atom
                    atom_in_stem = ring_stem_atom_dict[bond.begin_atom]
                    bond.begin_atom = atom_in_stem
                    atom_in_stem.out_bonds.append(bond)

                    if bond in pre_ring_atom.in_bonds:
                        pre_ring_atom.in_bonds.remove(bond)
                    else:
                        pre_ring_atom.out_bonds.remove(bond)
                elif bond.end_atom in ring_end_point:
                    pre_ring_atom = bond.end_atom
                    atom_in_stem = ring_stem_atom_dict[bond.end_atom]
                    bond.end_atom = atom_in_stem
                    atom_in_stem.in_bonds.append(bond)

                    if bond in pre_ring_atom.in_bonds:
                        pre_ring_atom.in_bonds.remove(bond)
                    else:
                        pre_ring_atom.out_bonds.remove(bond)

            # 更新ring_atom_list
            for id in range(len(ring_atom_list)):
                ring_atom = ring_atom_list[id]
                if ring_atom in ring_stem_atom_dict.keys():
                    ring_atom_list[id] = ring_stem_atom_dict[ring_atom]

            # 重新渲染cur_ring, 计算x,y, 以根据相对位置获得分支的顺序
            rend(ring_atom_list[0], scale=100, rend_name=1)

            for cid in range(len(common_bond_in_stem)):
                common_bond_in_stem[cid].m_type = common_bond_in_stem[cid].m_type.replace('@', '')
            # 处理环相连的分支,根据branch_info进行正确处理
            all_common_atom = set(list([atom for bond in common_bond_in_stem for atom in [bond.begin_atom, bond.end_atom]])) # 公共边上的所有原子
            

            # 使用ring_branch_info获得目标信息
            for atom in ring_atom_list:
                if atom.ring_branch_info is not None and len(atom.ring_branch_info) > 0:
                    while atom.ring_branch_info:
                        info = atom.ring_branch_info.pop(0)
                        tgt_bond = [item[0] for item in bond_dict if item[1] == info][0]

                        if tgt_bond.begin_atom in all_common_atom:
                            pre_begin_atom = tgt_bond.begin_atom
                            tgt_bond.begin_atom = atom
                            atom.out_bonds.append(tgt_bond)
                            pre_begin_atom.out_bonds.remove(tgt_bond)
                        elif tgt_bond.end_atom in all_common_atom:
                            pre_begin_atom = tgt_bond.end_atom
                            tgt_bond.end_atom = atom
                            atom.in_bonds.append(tgt_bond)
                            pre_begin_atom.in_bonds.remove(tgt_bond)
                        elif tgt_bond in intermediate_branch.keys(): # bond位于intermediate_branch
                            pre_begin_atom = intermediate_branch[tgt_bond]
                            if pre_begin_atom == tgt_bond.begin_atom:
                                tgt_bond.begin_atom = atom
                                atom.out_bonds.append(tgt_bond)
                                pre_begin_atom.out_bonds.remove(tgt_bond)
                            else:
                                tgt_bond.end_atom = atom
                                atom.in_bonds.append(tgt_bond)
                                pre_begin_atom.in_bonds.remove(tgt_bond)
                        intermediate_branch.pop(tgt_bond)   

            for bond in intermediate_branch.keys():
                # 将合并中间过程的这些bond全部添加到ring_atom_list[0]上
                cur_atom = ring_atom_list[0]
                tgt_bond = bond

                if tgt_bond.begin_atom == intermediate_branch[tgt_bond]:
                    pre_begin_atom = tgt_bond.begin_atom
                    tgt_bond.begin_atom = cur_atom
                    cur_atom.out_bonds.append(tgt_bond)
                    pre_begin_atom.out_bonds.remove(tgt_bond)
                elif tgt_bond.end_atom == intermediate_branch[tgt_bond]:
                    pre_begin_atom = tgt_bond.end_atom
                    tgt_bond.end_atom = cur_atom
                    cur_atom.in_bonds.append(tgt_bond)
                    pre_begin_atom.in_bonds.remove(tgt_bond)
                intermediate_branch[tgt_bond] = cur_atom

            if circle_bond: # 如果circle_bond连到common_bond上
                tgt_atom = get_index_atom_in_ring(ring_atom_list, circle_bond.branch_info.pop(0))
                if circle_atom == circle_bond.begin_atom:
                    pre_begin_atom = circle_bond.end_atom
                    circle_bond.end_atom = tgt_atom
                    pre_begin_atom.in_bonds.remove(circle_bond)
                    tgt_atom.in_bonds.append(circle_bond)
                elif circle_atom == circle_bond.end_atom:
                    pre_begin_atom = circle_bond.begin_atom
                    circle_bond.begin_atom = tgt_atom
                    pre_begin_atom.out_bonds.remove(circle_bond)
                    tgt_atom.out_bonds.append(circle_bond)
                else:
                    raise ValueError
            # merge_token_stack.pop(0)
            # 更新stem_all_atoms
            stem_all_atoms = GetAllAtoms(root_atom)
            if show:
                cv2.imwrite('rend_name.jpg', rend(stem_all_atoms[0], scale=100, rend_name=1))
            
        else:
            raise ValueError

    # --------------------------step3: 根据完全图生成chemfig -------------------
    stem_all_atoms = GetAllAtoms(stem_all_atoms[0])
    stem_all_atoms = sort_ring_atoms(stem_all_atoms)
    ssml_string = rend_text_with_branch_info(stem_all_atoms[0])
    ssml_string = ssml_string[0]

    return ssml_string


def cs_main(input_chemfig: str, is_show = False, need_ring_num = False):
    Bond.index = 0
    Atom.index = 0 # 重置分子id编号
    input_chemfig = input_chemfig.replace('branch(', '(').replace('branch)', ')')
    # input_chemfig = input_chemfig.replace('=^', '=').replace('=_', '=') # SSML中=和=^相同
    # input_chemfig = split_chemfig(input_chemfig)

    # 经过graph工具预处理,确保最终生成的string和原始的string完全相同
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
    input_chemfig = preprocess_chemfig

    # 预处理结束
    if need_ring_num:
        cs_string, branch_info, ring_branch_info, cond_data, total_ring_num = chemfig2chemstem(input_chemfig, show=is_show, add_extra_token = True, need_ring_num = need_ring_num)
    else:
        cs_string, branch_info, ring_branch_info, cond_data = chemfig2chemstem(input_chemfig, show=is_show, add_extra_token = True, need_ring_num = need_ring_num)

    return_cs_string = copy.deepcopy(cs_string)
    return_branch_info = copy.deepcopy(branch_info)
    return_ring_branch_info = copy.deepcopy(ring_branch_info)
    return_cond_data = copy.deepcopy(cond_data)
    
    # 使用ring_branch_info与cond_data还原
    generate_chemfig = chemstem2chemfig(cs_string, [None]*len(cs_string), ring_branch_info, cond_data, show=is_show, add_extra_token=True)
    
    # 图结构匹配
    is_same = True
    for i_unit, g_unit in zip(process_cond_render(input_chemfig, preprocess=False), process_cond_render(generate_chemfig, preprocess=False)):
        if not isinstance(i_unit, Atom):
            # 非分子部分直接并入out_units
            if i_unit != g_unit:
                is_same = False
        else:
            # 化学分子
            if match_graph(i_unit, g_unit): # match_graph返回值是相反的
                if is_show:
                    cv2.imwrite('rend_name.jpg', rend(i_unit, scale=100, rend_name=1))
                    cv2.imwrite('rend_name1.jpg', rend(g_unit, scale=100, rend_name=1))
                is_same = False
    if is_same:
        if need_ring_num:
            return True, return_cs_string, return_branch_info, return_ring_branch_info, return_cond_data, total_ring_num
        else:
            return True, return_cs_string, return_branch_info, return_ring_branch_info, return_cond_data
    else:
        if is_show:
            print("图匹配失败!!!!")
        return False, None, None, None, None


if __name__ == '__main__':
    str0 = '\chemfig { H _ { 2 } N -[:315] ?[a] branch( -[:0] -[:300] -[:240] branch( -[:180] -[:120] ?[a,{-}] branch( -:[:0] \circle branch) branch) -[:300] -[:345] ?[b] branch( -[:45] -[:345] branch( -[:45] O -[:345] ?[c] branch( -:[:0] \circle branch) branch( -[:45] branch( -[:120] I branch) -[:345] -[:285] branch( -[:225] -[:165] ?[c,{-}] -[:240] I branch) -[:345] -[:45] branch( <:[:105] H _ { 2 } N branch) -[:345] branch( -[:45] O H branch) =[:300] O branch) branch) -[:285] -[:225] -[:165] ?[b,{-}] branch( -:[:45] \circle branch) -[:225] H O branch) branch) }'
    str1 = '\chemfig { H O -[:0] ?[a] ( -[:60] -[:0] -[:300] ( -[:0] C O O H ) -[:240] -[:180] ?[a,{-}] -:[:60] \circle ) } + H B r'
    str2 = '\chemfig { H _ { 2 } N -[:345] ?[a] ( -:[:0] \circle ) ( -[:60] -[:0] -[:300] ( -[:240] -[:180] ?[a,{-}] ) -[:345] -[:45] ?[b] ( -[:0] ?[c] ( -[:45] ( -[:120] -[:195] ?[b,{-}] ) -[:345] -[:285] -[:225] -[:165] ?[c,{-}] ( -:[:45] \circle ) ) ) ) }'
    str3 = 'H B r + \chemfig { ?[a] -[:330] -[:30] -[:90] ( =[:45] ?[b] ( -[:0] ?[c] ( -:[:0] \circle ) ( -[:60] ( -[:0] -[:300] -[:240] -[:180] ?[c,{-}] ) -[:135] \Chemabove { N } { H } -[:210] ?[b,{-}] =[:150] O ) ) ) -[:165] \Chemabove { N } { H } ?[a,{-}] }'
    str4 = '\chemfig { ?[a] branch( -:[:0] \circle branch) -[:60] -[:0] N -[:300] branch( -[:0] N ?[b] branch( -[:315] -[:0] -[:60] N branch( -[:0] branch( -[:60] \Chemabove { N } { H } -[:0] N =[:60] -[:0] ?[c] branch( -:[:0] \circle branch) branch( -[:60] -[:0] -[:300] branch( -[:0] O H branch) -[:240] -[:180] ?[c,{-}] branch) branch) =[:300] O branch) -[:120] -[:180] ?[b,{-}] branch) branch) -[:240] -[:180] ?[a,{-}] }'
    str5 = '\chemfig { ?[a] -[:300] -[:15] ( -[:105] ?[a,{-}] ) -[:330] -[:30] S ( -[:30] -[:330] -[:30] ?[b] ( -:[:30] \circle ) ( -[:90] -[:30] -[:330] ( -[:30] O -[:330] -[:30] ( -[:90] O H ) -[:330] -[:30] \Chemabove { N } { H } -[:330] ( -[:30] ) -[:270] ) -[:270] -[:210] ?[b,{-}] ) ) ( =[:120] O ) =[:300] O }'
    str6 = '\chemfig { -[:15] ( -[:90] \Chemabove { N } { H } -[:15] ( -[:90] ?[a] ( -[:30] ( -[:90] -[:150] -[:210] -[:270] ?[a,{-}] ( -:[:30] \circle ) ) -[:315] N ( -[:15] S ( =[:15] O ) ( -[:105] ) =[:285] O ) -[:255] -[:300] ?[b] ( -[:0] -[:300] -[:240] -[:180] -[:120] ?[b,{-}] ( -:[:0] \circle ) ) ) ) =[:315] O ) -[:315] }'
    str7 = r'\chemfig { N ?[a] ( -:[:30] \circle ) -[:90] -[:30] -[:330] ( -[:270] -[:210] ?[a,{-}] ) -[:30] N ?[b] ( -[:345] -[:30] -[:90] N ( -[:150] -[:210] ?[b,{-}] ) -[:30] -[:330] ?[c] ( -[:30] N -[:315] ( -[:240] \Chemabove { H } { N } -[:165] ?[c,{-}] ( -:[:45] \circle ) ) -[:0] ?[d] ( -:[:0] \circle ) ( -[:60] -[:0] -[:300] -[:240] -[:180] ?[d,{-}] ) ) ) }'
    str8 = r"\chemfig { ?[a] ( -:[:30] \circle ) -[:90] -[:30] -[:330] ( -[:30] =[:345] -[:30] ( -[:345] ?[b] ( -[:30] -[:330] -[:270] -[:210] -[:150] ?[b,{-}] ( -:[:30] \circle ) ) ) -[:90] O O C C H _ { 3 } ) -[:270] -[:210] ?[a,{-}] } \underset { 手 \unk 催 化 剂 } { \overset { \chemfig { H _ { 3 } C -[:45] ( =[:90] O ) -[:315] -[:45] ( =[:90] O ) -[:315] O C H _ { 3 } } } { \rightarrow } } \chemfig { ?[a] ( -:[:30] \circle ) -[:90] -[:30] -[:330] ( -[:30] =[:345] -[:30] ( -[:345] ?[b] ( -[:30] -[:330] -[:270] -[:210] -[:150] ?[b,{-}] ( -:[:30] \circle ) ) ) -[:90] ( -[:30] ( -[:345] O C H _ { 3 } ) =[:90] O ) -[:150] ( -[:225] H _ { 3 } C ) =[:90] O ) -[:270] -[:210] ?[a,{-}] } + H O O C C H _ { 3 }" # 中文 + 化学方程式
    str9 = r'\chemfig { ?[a] ( -:[:30] \circle ) -[:90] -[:30] -[:330] -[:270] ( -[:210] ?[a,{-}] ) -[:315] ( -[:0] C ( -[:0] O -[:45] -[:315] ) =[:90] O ) -[:225] } \underset { H C l } { \overset { \chemfig { O ?[a] -[:45] -[:330] O -[:255] -[:180] ?[a,{-}] } } { \rightarrow } } \chemfig { ( -[:0] ?[a] ( -[:30] -[:330] -[:270] ( -[:210] -[:150] ?[a,{-}] ( -:[:30] \circle ) ) -[:315] ( -[:0] C ( -[:0] O -[:45] -[:315] ) =[:90] O ) -[:225] ) ) -[:90] C l }' # 
    str10 = '\chemfig { -[:330] ?[a] ( -[:15] S -[:300] ( -[:0] ( =[:60] O ) -[:300] N ?[b] ( -[:255] -[:300] -[:0] -[:60] -[:120] ?[b,{-}] ( -[:60] -[:0] \Chemabove { H } { N } -[:60] ( =[:0] O ) -[:120] H _ { 2 } N ) ) ) -[:240] -[:165] ?[a,{-}] ( -:[:30] \circle ) ) }' # 标注误差较大,导致渲染出原子相对位置发生改变
    str11 = '\\chemfig { ?[a] ( -:[:0] \\circle ) -[:60] ?[b] ( -[:0] ( -[:60] ?[c] ( -[:15] N ( -[:90] ( -[:0] S H ) -[:150] N -[:225] ?[c,{-}] ( -:[:0] \\circle ) -[:180] -[:240] N ?[b,{-}] ( -:[:0] \\circle ) ) -[:300] ?[d] ( -[:0] ( -[:60] ) -[:300] -[:240] -[:180] -[:120] ?[d,{-}] ( -:[:0] \\circle ) ) ) ) -[:300] -[:240] -[:180] ?[a,{-}] ) }' # 三个环相邻 train_02384.jpg
    str12 = '\chemfig { -[:285] -[:345] S -[:285] -[:345] -[:285] H N -[:345] ( =[:45] O ) -[:285] H N -[:345] ?[a] ( -:[:0] \circle ) ( -[:45] -[:345] ( -[:285] -[:225] -[:165] ?[a,{-}] -[:225] C l ) -[:45] ( =[:105] O ) -[:345] N ?[b] ( -[:45] -[:345] -[:285] O -[:225] -[:165] ?[b,{-}] ) ) }' # train_01878.jpg
    str13 = '\chemfig { ?[a] -[:45] -[:345] -[:285] ( -[:225] -[:165] ?[a,{-}] ( -:[:45] \circle ) ) -[:345] ( -[:45] N =[:345] N -[:45] ?[b] ( -:[:45] \circle ) ( -[:90] H N -[:30] ( -[:315] ?[c] ( -[:0] -[:300] -[:240] -[:180] -[:120] ?[c,{-}] ?[b,{-}] ( -:[:0] \circle ) ) ) -[:75] N H _ { 2 } ) ) =[:285] O }'
    str14 = '\chemfig { -[:0] ?[a] ( -[:30] ?[b] -[:285] -[:270] ( -[:30] ( -[:0] ) =[:90] ?[b,{-}] ) -[:150] ?[a,{=}] ) }'
    str15 = '\chemfig { -[:0] -[:300] -[:0] -[:300] -[:0] -[:300] N ?[a] -[:240] ?[b] ( -:[:15] \circle ) -[:195] -[:255] ( -:[:15] \circle ) -[:315] -[:15] -[:75] ?[b,{-}] ( -[:30] N -[:105] ?[a,{-}] ( -[:45] -[:345] S ( =[:75] O ) ( =[:255] O ) -[:345] ?[c] ( -:[:0] \circle ) ( -[:45] -[:345] -[:285] ( -[:225] -[:165] ?[c,{-}] ) -[:345] \Chemabove { H } { N } -[:45] ( -[:105] ) =[:345] O ) ) ) }'
    str16 = r"\chemfig { ?[a] =[:60] -[:0] ?[b] branch( -[:15] C H branch( -[:45] C H _ { 2 } I branch) -[:300] O -[:180] C branch( -[:180] ?[b,{=}] -[:240] =[:180] ?[a,{-}] branch) =[:270] O branch) } + N a O H \xrightarrow [ \triangle ] { 醇 } \chemfig { ?[a] =[:60] -[:0] ?[b] branch( -[:15] C branch( =[:45] C H _ { 2 } branch) -[:300] O -[:180] C branch( -[:180] ?[b,{=}] -[:240] =[:180] ?[a,{-}] branch) =[:270] O branch) } + N a I + H _ { 2 } O"
    str17 = '\chemfig { ?[a] -[:330] N ?[b] ( -[:105] -[:75] -[:45] ( -[:330] ( -[:255] ?[b,{-}] ) -[:15] \Chemabove { N } { H } -[:315] ( -[:15] ?[c] ( -:[:15] \circle ) ( -[:75] -[:15] -[:315] -[:255] ?[d] -[:195] ?[c,{-}] -[:270] N ( -:[:30] \circle ) -[:345] ( -[:45] N H ?[d,{-}] ) -[:285] -[:345] N ?[e] ( -[:300] -[:345] -[:45] N ( -[:105] -[:165] ?[e,{-}] ) -[:345] ?[f] ( -:[:0] \circle ) ( -[:45] ( -[:345] -[:285] -[:225] -[:165] ?[f,{-}] ) -[:105] O -[:45] ( -[:105] ) -[:345] ) ) ) ) =[:255] O ) -[:195] ?[a,{-}] ) }' # 具有嵌套环1
    str18 = '\chemfig { ?[a] -[:345] -[:30] ?[b] branch( -[:60] N branch( -[:0] branch( =[:60] O branch) -[:300] O -[:0] ?[c] branch( -:[:0] \circle branch) branch( -[:60] -[:0] -[:300] branch( -[:0] O -[:300] ?[d] branch( -[:0] -[:300] -[:240] -[:180] -[:120] ?[d,{-}] branch( -:[:0] \circle branch) branch) branch) -[:240] -[:180] ?[c,{-}] branch) branch) -[:120] -[:180] -[:240] N ?[a,{-}] branch( -[:30] -[:330] ?[b,{-}] branch) branch) }' # 具有嵌套环2
    str19 = '\chemfig { ?[a] -[:345] -[:45] branch( -[:105] -[:180] ?[a,{-}] branch) -[:345] N ?[b] branch( -[:300] branch( >:[:345] ?[c] branch( -[:45] N branch( -[:0] -[:285] ?[d] branch( -[:225] branch( -:[:0] \circle branch) branch( -[:165] ?[c,{-}] branch) -[:285] -[:345] -[:45] branch( -[:105] ?[d,{-}] branch) -[:345] O -[:285] -[:345] ?[e] branch( -:[:0] \circle branch) branch( -[:45] -[:345] branch( -[:285] -[:225] -[:165] ?[e,{-}] branch) -[:45] F branch) branch) branch) -[:105] branch( -[:165] ?[b,{-}] branch) =[:45] O branch) branch) =[:225] O branch) }'
    str20 = '\chemfig { ?[a] -[:285] -[:330] -[:15] -[:75] -[:120] branch( -[:75] \Chemabove { N } { H } -[:30] branch( >:[:75] ?[b] branch( -[:225] -[:150] -[:90] -[:30] branch( -[:45] branch) branch( -[:135] branch) >:[:330] ?[b,{-}] branch( -[:30] -[:330] -[:270] ?[c] -[:210] ?[b,{-}] -[:270] branch( -:[:30] \circle branch) branch( -[:240] H O branch) -[:330] branch( -[:30] branch( -[:90] ?[c,{-}] branch) -[:330] branch( -[:30] branch) -[:270] branch) -[:270] O H branch) branch) branch) =[:285] O branch) -[:180] ?[a,{-}] }'
    str21 = r'\chemfig { ?[a] =[:90] -[:30] =[:330] branch( -[:30] =[:300] branch) -[:270] =[:210] ?[a,{-}] -[:270] N O _ { 2 } } + N a O H \xrightarrow [ \triangle ] { 乙 醇 } \chemfig { ?[a] =[:90] -[:30] =[:330] branch( -[:30] C ~ C H branch) -[:270] =[:210] ?[a,{-}] -[:270] N O _ { 2 } } + N a B r + H _ { 2 } O'
    str22 = '\chemfig { -[:30] O -[:75] -[:30] -[:75] N ?[a] branch( -[:15] ?[b] branch( -[:270] -[:315] -[:15] branch( -[:75] -[:135] ?[b,{-}] branch) -[:315] \Chemabove { H } { N } -[:15] -[:315] ?[c] branch( -[:30] -[:330] -[:270] branch( -[:210] -[:150] ?[c,{-}] branch( -:[:30] \circle branch) branch) -[:315] O -[:15] branch) branch) -[:75] -[:150] -[:225] ?[a,{-}] =[:165] O branch) }'
    str23 = "\chemfig { >[:60] ?[a] ( -[:0] O -[:60] ?[b] ( -[:60] ( -[:0] =[:300] -[:240] ( -[:180] O ?[b,{-}] ) <:[:300] O H ) =[:120] O ) ( >:[:120] -[:180] -[:240] ?[a,{-}] ) ) }"
    str24 = "\chemfig { <[:0] ?[a] ( -[:45] ( -[:105] ) -[:330] ( >[:30] -[:330] ( -[:30] ) -[:270] ) ( -[:120] O -[:195] O ?[a,{-}] ) -[:270] O -[:195] O ?[a,{-}] ) }" # 公共边大于1
    str25 = "\chemfig { O =[:30] ?[a] ( -[:90] ( -[:30] ?[b] ( -[:30] -[:330] -[:270] -[:210] ( -[:150] ?[b,{-}] ) ( -[:210] O ?[a,{-}] ) <[:270] H ) ( <[:150] H ) ) =[:150] ) }" # 公共边大于1
    str26 = "\chemfig { -[:300] ( -[:0] ?[a] ( -[:15] O -[:0] O -[:0] ( -[:0] -[:300] O H ) ( -[:120] -[:180] ?[a,{-}] ) -[:240] =[:180] ?[a,{-}] ) ) -[:240] }" # 死循环
    str27 = "\chemfig { ?[a] -[:60] N ?[b] ( -[:0] -[:300] -[:240] ( -[:120] ( -[:15] ?[c] ( -[:60] O -[:345] N =[:270] ( -[:210] N ?[c,{=}] ) -[:330] ) ) -[:120] ?[b,{-}] ) -[:180] ?[a,{-}] ) }"


    # str28 = "\chemfig { N ?[a] -[:45] O -[:345] ( -[:30] =[:345] ?[b] ( -[:90] ?[c] ( -[:210] -[:270] -[:330] N ( -[:30] -[:90] ?[c,{-}] ) -[:90] ?[b,{-}] ) ) ) =[:270] -[:195] ?[a,{=}] }"
    # str29 = "\chemfig { ?[a] -[:60] ?[b] ( -[:0] -[:300] -[:240] ( -[:120] N ( -[:15] ?[c] ( -[:60] N =[:0] -[:300] ( -[:0] B r ) =[:240] -[:180] ?[c,{=}] ) ) -[:120] O ?[b,{-}] ) -[:180] ?[a,{=}] ) }"
    
    need_ring_num=True
    input_str = str19
    success, cs_string, _, ring_branch_info, cond_data = cs_main(str1, is_show=True) # str14
    if success:
        print("Success!")
    else:
        print("False!")
    input()
    
    test_list = [str0, str1, str2, str3, str4, str5, str6, str7, str8, str9, str10, str11, str12, str13, str14, str15, str16, str17, str18, str19, str20, str21, str22, str23, str24, str25, str26, str27]
    # test_list = [str26, str27]
    for tid in range(len(test_list)):
        try:
            if tid == 26:
                pass
            success, cs_string, branch_info, ring_branch_info, cond_data = cs_main(test_list[tid], is_show=False)
            if success:
                print(str(tid) + " Success")
            else:
                print(str(tid) + " False!!!")
        except Exception as e:
            print(tid, "error", e)


