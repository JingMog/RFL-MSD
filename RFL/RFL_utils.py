# @Time      : 2024/3/11

from chemfig_struct import Atom, Bond, CircleAtom
from cond_render import GetAllAtoms, chemfig_random_cond_parse
from chemfig_ops import GetCoordRange
from text_render import dfs_visited
from utils import replace_chemfig
from Tool_Formula.latex_norm.transcription import parse_transcription
from image_render import rend
from chemfig_ssml_struct import bond_types

import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import copy
from collections import Counter

def BFS_visit(start_atom: Atom, show=False):
    '''
        通过BFS遍历图,同时保存child_table,之后根据child_table回溯获得最小环的路径
        目前的BFS可能无法处理环的嵌套问题,例如六边形中间有两个键将环分成两部分,这种情况在后处理部分仍然将其视为一个环. 
        2024.06.28:已经被DFS淘汰
    '''
    def path_back_track(child_table, cur_atom):
        '''BFS遍历图中, 根据child_table回溯获得环的路径'''
        visited = []
        
        path_backtrack_queue = [cur_atom]
        path = []
        while len(path_backtrack_queue) > 0:
            front = path_backtrack_queue.pop(0)
            if front in visited:
                break
            path.append(front)
            visited.append(front)
            for key, value in child_table.items():
                for v in value:
                    if v == front:
                        path_backtrack_queue.append(key)
        
        if len(path) > 0:
            neighbor_num_list = []
            for item in path:
                neighbor_num = 0 # 与环内相邻的原子数
                for n in item.all_bonds:
                    if n[2] in path:
                        neighbor_num += 1
                neighbor_num_list.append(neighbor_num)
            
            while 1 in neighbor_num_list:
                neighbor_num_list = []
                for item in path:
                    neighbor_num = 0 # 与环内相邻的原子数
                    for n in item.all_bonds:
                        if n[2] in path:
                            neighbor_num += 1
                    neighbor_num_list.append(neighbor_num)
                # 检测BFS回路上原子是否两两相邻,删除多余分支的节点
                for index in range(len(neighbor_num_list)-1, -1, -1): # 倒叙遍历
                    if neighbor_num_list[index] < 2:
                        path.pop(index)
                
        return path

    def get_bonds_from_ring_atoms(atoms_list):
        '''根据环上的atom获取环的所有bond'''
        # 先获取所有bond, 然后将不在环上的bond去除
        all_bond_list = [bond for atom in atoms_list for bond in atom.in_bonds + atom.out_bonds]
        real_bond_list = []
        for bond in all_bond_list:
            if bond.begin_atom in atoms_list and bond.end_atom in atoms_list:
                real_bond_list.append(bond)
        real_bond_list = set(list(real_bond_list))
        return real_bond_list

    visited = []
    child_table = {} # 记录节点扩展顺序,以便回溯路径
    ring_id = 0 # 记录ring的id
    ring_paths = {} # 最小环的路径
    atom_queue = [start_atom] # 第一个原子入队
    while len(atom_queue) > 0: # 当队列不为空
        front = atom_queue.pop(0)
        if front in visited:
            # 根据child_table回溯路径
            atom_path = path_back_track(child_table, front) # front必为环上的节点
            ring_id += 1
            ring_paths[ring_id] = atom_path
            continue # 检测到环,说明该节点已经被扩展过,不需要继续扩展

        visited.append(front)
        child_list = []
        for bond in front.all_bonds: # 扩展临边
            next_atom = bond[2]
            if next_atom != front and next_atom not in visited: #and next_atom not in atom_queue:
                child_list.append(next_atom)
                atom_queue.append(next_atom)
            child_table[front] = child_list
    
    # 2024.06.27 new
    if len(ring_paths) > 1:
        tmp_remove_list = [] # [内部环i, 包裹的环j]
        for i in range(len(ring_paths)):
            for j in range(i+1, len(ring_paths)):
                keys = list(ring_paths.keys())
                # 如果两环有交集,就判断是否发生环的嵌套
                common_elements = set(ring_paths[keys[i]]).intersection(ring_paths[keys[j]]) # 返回两个集合的交集
                if len(common_elements) > 0:
                    is_nest, nest_id = judge_if_ring_nest(ring_paths[keys[i]], ring_paths[keys[j]])
                    bonds_i = get_bonds_from_ring_atoms(ring_paths[keys[i]])
                    bonds_j = get_bonds_from_ring_atoms(ring_paths[keys[j]])
                    common_bonds = get_bonds_from_ring_atoms(common_elements)
                    if is_subset(ring_paths[keys[i]], ring_paths[keys[j]]):
                        # ring[j]包括两环的全部, ring[i]为子集
                        print("发生嵌套!")
                        raise ValueError("发生嵌套")

                    elif is_subset(ring_paths[keys[j]], ring_paths[keys[i]]):
                        # ring[i]包括两环的全部, ring[j]为子集
                        print("发生嵌套!")
                        raise ValueError("发生嵌套")

                    elif is_nest and nest_id==0:
                        # ring[j]包裹着ring[i]
                        real_common_bonds = bonds_i - common_bonds # 视觉上的公共边
                        bonds_i = bonds_i # 小环不用变
                        bonds_j = bonds_j - common_bonds + real_common_bonds
                    elif is_nest and nest_id==1:
                        # ring[i]包裹着ring[j]
                        real_common_bonds = bonds_j - common_bonds
                        bonds_j = bonds_j # 小环不用变
                        bonds_i = bonds_i - common_bonds + real_common_bonds
                    
        if tmp_remove_list:
            tmp_remove_list = sorted(tmp_remove_list, reverse=True)
            for remove_unit in tmp_remove_list:
                ring_paths[remove_unit[1]] += ring_paths[remove_unit[0]] # 需要将remove_unit[0]合并到remove_unit[1]中
                ring_paths[remove_unit[1]] = list(set(ring_paths[remove_unit[1]]))
                ring_paths.pop(remove_unit[0])
            
            # 重新编号生成ring_paths
            values = list(ring_paths.values())
            ring_id = 0
            ring_paths = {}
            for v in values:
                ring_id += 1
                ring_paths[ring_id] = v

    if show:
        print("环数量:", ring_id)
    
    return ring_paths, ring_id


def DFS_visit(start_atom: Atom, show=False):
    '''使用DFS检测所有的环结构, 并删除嵌套的环来获得所有的简单环路'''
    def dfs(node, start, visited, path):
        visited[node] = True
        path.append(node)
        for neighbor in node.all_bonds:
            neighbor = neighbor[2]
            if not visited[neighbor]:
                dfs(neighbor, start, visited, path)
            elif neighbor == start and len(path) > 2:
                cycle = path[:]  # Make a copy of the path
                cycle.append(start)
                cycles.append(cycle)
        path.pop()
        visited[node] = False
    
    def remove_duplicate_cycles(cycles):
        '''移除重复的环'''
        unique_cycles = []
        seen = set()
        for cycle in cycles:
            cycle = list(set(cycle))
            sorted_cycle = tuple(sorted(cycle, key=lambda x: x.name))
            if sorted_cycle not in seen:
                unique_cycles.append(cycle)
                seen.add(sorted_cycle)
        return unique_cycles

    def remove_nest_cycles(cycles):
        '''移除嵌套的环'''
        remove_id = []
        for i in range(len(cycles)):
            for j in range(i+1, len(cycles)):
                is_nest, nest_id = judge_if_ring_nest(cycles[i], cycles[j])
                if is_nest and (nest_id == 0) and (j not in remove_id): # cycles[j]包裹着cycles[i]
                    remove_id.append(j)
                if is_nest and (nest_id == 1) and (i not in remove_id): # cycles[i]包裹着cycles[j]
                    remove_id.append(i)
                elif is_subset(cycles[i], cycles[j]) and (j not in remove_id): # ring[j]包括两环的全部, ring[i]为子集
                    remove_id.append(j)
                elif is_subset(cycles[j], cycles[i]) and (i not in remove_id):
                    remove_id.append(i)
                else:
                    pass
        simple_cycles = []
        remove_id = list(set(remove_id))
        for i in range(len(cycles)):
            if i not in remove_id:
                simple_cycles.append(cycles[i])
        return simple_cycles

    # DFS检测图中的所有环路
    nodes = GetAllAtoms(start_atom)
    cycles = []
    for node in nodes:
        visited = {n: False for n in nodes}
        path = []
        dfs(node, node, visited, path)

    # Remove duplicate cycles
    unique_cycles = remove_duplicate_cycles(cycles) # 删去重复的环路
    unique_cycles = remove_nest_cycles(unique_cycles) # 删除嵌套的环路, 只保留简单环路
    ring_num = len(unique_cycles)
    # 将cycles转化为字典,保证接口一致
    ring_paths = {i+1: unique_cycles[i] for i in range(ring_num)}
    
    if show:
        print("环数量:", ring_num)
    return ring_paths, ring_num


def judge_if_ring_nest(ring_i, ring_j):
    '''判断两环是否发生嵌套'''
    # 构建连通域
    polygon_i = []
    polygon_j = []
    for item_i in ring_i: # TODO:多边形的点要按照顺序
        polygon_i.append([int(item_i.pos_x * 100), -int(item_i.pos_y * 100)])
    for item_j in ring_j:
        polygon_j.append([int(item_j.pos_x * 100), -int(item_j.pos_y * 100)])
    polygon_i = np.array(polygon_i)
    polygon_j = np.array(polygon_j)
    # 找到顶点的凸包,确保正确构建多边形
    polygon_i = cv2.convexHull(polygon_i)
    polygon_i = polygon_i.reshape((-1, 2))
    polygon_j = cv2.convexHull(polygon_j)
    polygon_j = polygon_j.reshape((-1, 2))

    # 判断点是否在多边形内部,在内部pointPolygonTest返回+1
    # 判断j是否在i内
    for atom_i in ring_i:
        if atom_i in ring_j: # 公共原子跳过
            continue
        point_i = (int(atom_i.pos_x * 100), -int(atom_i.pos_y*100))
        # draw_polygon_point(polygon_j, point_i)
        if cv2.pointPolygonTest(polygon_j, point_i, False) > 0:
            return True, 0
    # 判断i是否在j内
    for atom_j in ring_j:
        if atom_j in ring_i: # 公共原子跳过
            continue
        point_j = (int(atom_j.pos_x * 100), -int(atom_j.pos_y*100))
        # draw_polygon_point(polygon_i, point_j)
        if cv2.pointPolygonTest(polygon_i, point_j, False) > 0:
            return True, 1

    return False, None

def draw_polygon_point(polygon, point):
    '''绘制多边形以及点'''
    fig, ax = plt.subplots()
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])
    ax.set_aspect("equal")
    ax.add_patch(plt.Polygon(polygon, fill=False))
    
    pt = point
    color = "green"
    ax.scatter(pt[:1], pt[1:], color=color)
    ax.text(*pt, "({}, {})".format(*pt), fontsize=12)
    
    plt.savefig('./polygon.jpg', dpi=150)





def get_ring_bond_dict(ring_paths):
    '''获取环内有所的bond'''
    ring_bond_dict = {} # 记录环内所有的bond
    for id, atom_list in ring_paths.items():
        ring_bond_dict[id] = []
        for i in range(len(atom_list)):
            for j in range(i + 1, len(atom_list)):
                bond = get_bond_between_2atom(atom_list[i], atom_list[j])
                if bond:
                    ring_bond_dict[id].append(bond)
    # print("每个环内的键:", ring_bond_dict)
    return ring_bond_dict

def parse_ring_adj_relation(ring_paths, ring_id):
    '''根据ring_paths解析环之间的相邻关系'''
    # 解析环之间的相邻关系
    # 获得所有环之后,解析每个环相邻边的数量,从0相邻的环开始处理
    adj_ring_info = {} # 具体的相邻关系,   (ring_i, ring_j):公共原子
    adj_ring_bond = {} # 相邻关系, (ring_i, ring_j):公共边
    adj_ring_num = {} # 相邻的环数量, ring_id:相邻环数
    if ring_id > 1: # 两个以上环时解析环关系
        for ring_id_i, ring_path_i in ring_paths.items():
            adj_atom_num = 0
            for ring_id_j, ring_path_j in ring_paths.items():
                if ring_id_j > ring_id_i: # 判断和其他环共同原子的数量
                    common_atom = list(set(ring_path_i) & set(ring_path_j))

                    # 获取common_bond
                    adj_ring_bond[(ring_id_i, ring_id_j)] = []
                    for i in range(len(common_atom)): # 两个环之间有多条公共边的情况
                        for j in range(i+1, len(common_atom)):
                            common_bond = get_bond_between_2atom(common_atom[i], common_atom[j])
                            if common_bond:
                                adj_ring_bond[(ring_id_i, ring_id_j)].append(common_bond)

                    adj_ring_info[(ring_id_i, ring_id_j)] = common_atom
                    #桥环烷烃即公共原子超过两个

        for ring_id_i in ring_paths.keys():
            num = 0
            for info_id, info in adj_ring_info.items():
                if (ring_id_i in info_id) and (len(info) > 1): # 只有原子相邻不算两个环相邻
                    num += 1
            adj_ring_num[ring_id_i] = num
        
    elif ring_id == 1: # 只有一个环
        adj_ring_info[(ring_id, ring_id)] = []
        adj_ring_num[ring_id] = 0
        adj_ring_bond[(ring_id, ring_id)] = []
    
    return adj_ring_info, adj_ring_bond, adj_ring_num


def get_bond_between_2atom(atom1: Atom, atom2: Atom):
    '''获取两个原子之间的边'''
    bond1 = atom1.in_bonds + atom1.out_bonds
    bond2 = atom2.in_bonds + atom2.out_bonds
    common_bond = list(set(bond1) & set(bond2))
    if len(common_bond) > 0:
        return common_bond[0]
    else:
        return False


def bond_in_ring(bond, ring_path):
    '''判断bond是否在环路径ring_path内'''
    bond_list = []
    for i in range(len(ring_path)):
        for j in range(i+1, len(ring_path)):
            bond = get_bond_between_2atom(ring_path[i], ring_path[j])
            bond_list.append(bond)
    
    print(bond)
    return False

def sort_ring_atoms(atom_list):
    '''按照特定规则对环内的原子进行排序'''
    rend(atom_list[0], rend_name=1, scale=100)
    atom_list = sorted(atom_list, key=lambda x: round(x.pos_x, 2) * 10000 + round(x.pos_y, 2)) # 选择最左上角的原子作为begin_atom
    begin_atom = atom_list[0]
    cur_atom = begin_atom
    visited = [] # 从begin_atom开始逆时针遍历得到顺序
    while True:
        if cur_atom in visited:
            break

        visited.append(cur_atom)
        another_atom_list = []
        for bond in cur_atom.in_bonds + cur_atom.out_bonds:
            if bond.begin_atom == cur_atom:
                another_atom = bond.end_atom
            else:
                another_atom = bond.begin_atom
            if another_atom in atom_list and another_atom not in visited and another_atom.m_text != '\circle':
                another_atom_list.append(another_atom)
        another_atom_list = sorted(another_atom_list, key=lambda x: - (x.pos_x - x.pos_y) ) # 找到与cur_atom相邻的最右上角的原子,x大,y小
        if len(another_atom_list) == 0:
            break
        cur_atom = another_atom_list[0]
    
    return visited

def get_atom_index_in_ring(atom_list, atom):
    '''对atom_list进行排序,得到atom在排序后列表的位置'''
    sorted_atom_list = sort_ring_atoms(atom_list)
    index = sorted_atom_list.index(atom)

    return index

def get_index_atom_in_ring(atom_list, index):
    '''对atom_list进行排序,得到索引为index的原子'''
    sorted_atom_list = sort_ring_atoms(atom_list)
    atom = sorted_atom_list[index]
    return atom


def get_ring_adj_branch_info(ring_paths, del_other_ring=False):
    '''获取所有环相连的分支信息'''
    # 初始化
    ring_adj_bond = {} # 记录每个环相连的键,后续还需要将这些键加到super_atom上
    for i in ring_paths.keys():
        ring_adj_bond[i] = []

    for ring_id_i, path in ring_paths.items():
        # 环内每个原子
        for atom in path:
            # 每个原子相连的所有键
            for bond in atom.in_bonds + atom.out_bonds:
                if not (bond.begin_atom in path and bond.end_atom in path):
                    # 所连的原子不在环内,说明该分支为环所相连的分支
                    ring_adj_bond[ring_id_i].append(bond)
    
    if del_other_ring:
        # 针对两环只有一个公共点
        other_ring_paths = [item for ring_bond in get_ring_bond_dict(ring_paths).values() for item in ring_bond]
        for ring_id, adj_bond in ring_adj_bond.items():
            new_adj_bond = []
            for bond in adj_bond:
                if bond in other_ring_paths:
                    continue
                new_adj_bond.append(bond)
            ring_adj_bond[ring_id] = new_adj_bond

    return ring_adj_bond

def get_common_bond(adj_ring_bond, ring_i, ring_j):
    '''获取环i与环j的公共边'''
    common_bond = []
    for key, value in adj_ring_bond.items():
        if (ring_i, ring_j) == key or (ring_j, ring_i) == key:
            common_bond += value

    assert len(common_bond) > 0

    return common_bond



def get_adj_bond(common_bond):
    '''获取common_bond相邻的键和对应的原子'''
    atom_list = [atom for bond in common_bond for atom in [bond.begin_atom, bond.end_atom]]
    atom_counts = Counter(atom_list)
    atom_end_point = [element for element, count in atom_counts.items() if count == 1] # 端点
    assert len(atom_end_point) == 2, "端点不能超过两个"
    another_bond_in_ring1 = None
    another_bond_in_ring2 = None
    for bond in atom_end_point[0].in_bonds + atom_end_point[0].out_bonds:
        if bond not in common_bond:
            another_bond_in_ring1 = bond
            break
    
    for bond in atom_end_point[1].in_bonds + atom_end_point[1].out_bonds:
        if bond not in common_bond:
            another_bond_in_ring2 = bond
            break

    return another_bond_in_ring1, another_bond_in_ring2

def select_tgt_atom_by_distance(cur_ring_path, common_bond, cur_bond):
    '''在边的归并过程中根据cur_bond与common_bond的距离远近选择tgt_atom'''
    common_atom1 = common_bond.begin_atom
    common_atom2 = common_bond.end_atom
    if cur_bond.begin_atom in cur_ring_path:
        cur_atom = cur_bond.begin_atom
    else:
        cur_atom = cur_bond.end_atom
    # cur_ring_path = sort_ring_atoms(cur_ring_path)
    # 计算common_atom1与cur_atom,common_atom2与cur_atom的距离
    # index1 = cur_ring_path.index(common_atom1)
    # index2 = cur_ring_path.index(common_atom2)
    # cur_index = cur_ring_path.index(cur_atom)
    distance1 = math.sqrt((round(common_atom1.pos_x, 2) - round(cur_atom.pos_x, 2))**2 + (round(common_atom1.pos_y,2) - round(cur_atom.pos_y,2))**2)
    distance2 = math.sqrt((round(common_atom2.pos_x,2) - round(cur_atom.pos_x,2))**2 + (round(common_atom2.pos_y,2) - round(cur_atom.pos_y,2))**2)
    if distance1 < distance2:
        return common_atom1
    elif distance1 > distance2:
        return common_atom2
    else:
        return common_atom1


def search_super_bond_in_ring(ring_atom_list):
    '''在单独的环中寻找Super_bond'''
    common_bond = []
    for atom in ring_atom_list:
        for bond in atom.in_bonds + atom.out_bonds:
            if '#' in bond.m_type:
                common_bond.append(bond)
                # common_bond.m_type = common_bond.m_type.replace('@', '') # 消除这个superbond
    common_bond = list(set(common_bond))
    return common_bond

def search_super_bond_in_stem(stem_all_atoms, common_bond_in_ring):
    '''根据common_bond_in_ring在骨干中找到对应的superbond'''
    # 先找到Superbond, 然后找剩余的公共边
    common_bond_in_stem = []
    stem_ring_bond_dict = {} # stem_bond: ring_bond
    ring_stem_atom_dict = {} # ring_atom: stem_atom (only common part)
    all_stem_bonds = [bond for atom in stem_all_atoms for bond in atom.in_bonds+atom.out_bonds]
    all_stem_bonds = list(set(all_stem_bonds))
    
    token_ring_bond = None # 记录有token标记的环上Bond
    token_stem_bond = None
    for bond in all_stem_bonds:
        for common_bond in common_bond_in_ring:
            if (bond.m_type == common_bond.m_type.replace('#', '@')) and (abs(bond.m_angle - common_bond.m_angle) <= 10 or (abs(bond.m_angle-common_bond.m_angle)>=165 and abs(bond.m_angle-common_bond.m_angle)<=195)): # bond_type相同, 角度相差0或180
                common_bond_in_stem.append(bond)
                stem_ring_bond_dict[bond] = common_bond
                token_stem_bond = bond
                token_ring_bond = common_bond
                if abs(bond.m_angle - common_bond.m_angle) < 30: # 方向一致
                    ring_stem_atom_dict[token_ring_bond.begin_atom] = token_stem_bond.begin_atom
                    ring_stem_atom_dict[token_ring_bond.end_atom] = token_stem_bond.end_atom
                else: # 方向相反
                    ring_stem_atom_dict[token_ring_bond.begin_atom] = token_stem_bond.end_atom
                    ring_stem_atom_dict[token_ring_bond.end_atom] = token_stem_bond.begin_atom
                break
        if common_bond_in_stem:
            break
    
    if len(common_bond_in_ring) > 1: # 有多条公共边
        unprocessed_bonds = [bond for bond in common_bond_in_ring if bond != token_ring_bond] # 记录未找到对应关系的换上bond
        cnum = 0
        while unprocessed_bonds:
            cnum += 1
            if cnum > 10:
                raise ValueError("陷入死循环")
            tmp_common_bonds = unprocessed_bonds
            for ring_common_bond in tmp_common_bonds:
                if ring_common_bond.begin_atom in ring_stem_atom_dict.keys() or ring_common_bond.end_atom in ring_stem_atom_dict.keys():
                    if ring_common_bond.begin_atom in ring_stem_atom_dict.keys(): # 环上的atom对应到主干中
                        stem_begin_atom = ring_stem_atom_dict[ring_common_bond.begin_atom] 
                    else:
                        stem_begin_atom = ring_stem_atom_dict[ring_common_bond.end_atom]
                    ring_begin_atom = ring_common_bond.begin_atom
                    
                    candidate_bonds = [bond for bond in all_stem_bonds if (bond.begin_atom == stem_begin_atom or bond.end_atom == stem_begin_atom) and (bond not in stem_ring_bond_dict.keys())] # 只在和stem_begin_atom相连的bond中搜索
                    for bond in candidate_bonds:
                        # print(bond)
                        if (bond.m_type == ring_common_bond.m_type.replace('#', '')) and abs(bond.m_angle - ring_common_bond.m_angle) < 20: # 键类型相同,  角度相差小于30度
                            ring_stem_atom_dict[ring_common_bond.begin_atom] = bond.begin_atom # 维护atom_dict
                            ring_stem_atom_dict[ring_common_bond.end_atom] = bond.end_atom
                            # if (bond.begin_atom == stem_begin_atom): # 方向相同,
                            #     ring_stem_atom_dict[ring_common_bond.begin_atom] = bond.begin_atom # 维护atom_dict
                            #     ring_stem_atom_dict[ring_common_bond.end_atom] = bond.end_atom
                            # elif (bond.end_atom == stem_begin_atom): # 方向相反
                            #     ring_stem_atom_dict[ring_common_bond.begin_atom] = bond.end_atom # 维护atom_dict
                            #     ring_stem_atom_dict[ring_common_bond.end_atom] = bond.begin_atom
                            common_bond_in_stem.append(bond)
                            stem_ring_bond_dict[bond] = ring_common_bond # 维护bond_dict
                            unprocessed_bonds.remove(ring_common_bond)
                            # print(ring_stem_atom_dict)
                            break
                        elif (bond.m_type == ring_common_bond.m_type.replace('#', '')) and abs(bond.m_angle - ring_common_bond.m_angle) > 160 and abs(bond.m_angle - ring_common_bond.m_angle) < 200:# 键类型相同, 方向相反, 角度相差在180附近
                            ring_stem_atom_dict[ring_common_bond.begin_atom] = bond.end_atom # 维护atom_dict
                            ring_stem_atom_dict[ring_common_bond.end_atom] = bond.begin_atom
                            # if (bond.begin_atom == stem_begin_atom):
                            #     ring_stem_atom_dict[ring_common_bond.begin_atom] = bond.begin_atom
                            #     ring_stem_atom_dict[ring_common_bond.end_atom] = bond.end_atom 
                            # elif (bond.end_atom == stem_begin_atom): # 方向相反
                            #     ring_stem_atom_dict[ring_common_bond.begin_atom] = bond.end_atom # 维护atom_dict
                            #     ring_stem_atom_dict[ring_common_bond.end_atom] = bond.begin_atom
                            common_bond_in_stem.append(bond)
                            stem_ring_bond_dict[bond] = ring_common_bond # 维护bond_dict
                            unprocessed_bonds.remove(ring_common_bond)
                            # print(ring_stem_atom_dict)
                            break


    assert len(common_bond_in_stem) > 0, "common_bond_in_stem不能为空"
    return common_bond_in_stem, ring_stem_atom_dict, stem_ring_bond_dict

def judge_point_in_polygon(atom_list, atom):
    '''判断点是否在多边形内'''
    polygon = []
    for item in atom_list:
        polygon.append([int(item.pos_x * 100), int(item.pos_y * 100)])
    polygon = np.array(polygon)
    point = (int(atom.pos_x * 100), int(atom.pos_y*100))

    # 判断点是否在多边形内部,内部返回+1
    if cv2.pointPolygonTest(polygon, point, False) > 0:
        return True
    else:
        return False

def is_subset(list1, list2):
    '''判断list1是否是list2的子集'''
    set1 = set(list1)
    set2 = set(list2)

    return set1.issubset(set2) # set1是否是set2的子集


def branch_info2str(branch_info):
    '''将branch_info转化为字符串'''
    branch_str = '.'.join(str(item) if item is not None else 'None' for item in branch_info)
    return branch_str


def str2branch_info(branch_str):
    '''将branch_info字符串恢复为列表'''
    branch_str = branch_str.split('.')
    branch_info = [eval(item) if item != 'None' else None for item in branch_str]
    return branch_info

def process_cs_string2units(cs_string, branch_info, ring_branch_info, cond_data):
    '''将cs_string根据分子划分为多个units,同步处理branch_info,ring_branch_info'''
    in_units = [] # cs_string字符串根据分子划分为多个units
    in_branch_info_units = []
    in_ring_branch_info_units = []
    in_cond_data_units = []

    stack = [] # 用于括号匹配划分分子
    index_stack = []
    begin_index = [] # 保存分子的begin_index
    end_index = [] # 保存分子的end_index
    for i in range(len(cs_string)):
        if cs_string[i] == '\chemfig':
            stack.append('\chemfig')
            index_stack.append(i)
        elif cs_string[i] == '}':
            stack.pop()
            e_index = index_stack.pop()
            if stack and stack[-1] == '\chemfig':
                end_index.append(i)
                stack.pop()
                b_index = index_stack.pop()
                begin_index.append(b_index)
        elif cs_string[i] == '{':
            stack.append('{')
            index_stack.append(i)
    i = 0
    # 根据匹配的结果划分分子
    while i < len(cs_string):
        if begin_index:
            cur_begin = begin_index[0]
            cur_end = end_index[0]
            if cur_begin <= i and i <= cur_end:
                in_units.append(cs_string[cur_begin: cur_end + 1])
                in_branch_info_units.append(branch_info[cur_begin: cur_end + 1])
                in_ring_branch_info_units.append(ring_branch_info[cur_begin: cur_end + 1])
                in_cond_data_units.append(cond_data[cur_begin: cur_end + 1])
                i = cur_end + 1
                begin_index.pop(0)
                end_index.pop(0)
            else:
                in_units.append(cs_string[i])
                in_branch_info_units.append(branch_info[i])
                in_ring_branch_info_units.append(ring_branch_info[i])
                in_cond_data_units.append(cond_data[i])
                i += 1
        # chemfig字符串已经遍历结束
        else:
            in_units.append(cs_string[i])
            in_branch_info_units.append(branch_info[i])
            in_ring_branch_info_units.append(ring_branch_info[i])
            in_cond_data_units.append(cond_data[i])
            i += 1

    assert len(in_units) == len(in_branch_info_units), "in_units必须和in_branch_info_units长度相同"
    assert len(in_units) == len(in_ring_branch_info_units), "in_units必须和in_ring_branch_info_units长度相同"
    assert len(in_units) == len(in_cond_data_units), "in_units必须和in_cond_data_units长度相同"

    return in_units, in_branch_info_units, in_ring_branch_info_units, in_cond_data_units



def rend_text_with_branch_info(rootAtom, withAngle=1, branch_de_amb=0, bond_dict=[]):
    all_atoms = GetAllAtoms(rootAtom)
    min_x, min_y, max_x, max_y = GetCoordRange(all_atoms)
    
    #select start
    anchor_x = min_x
    anchor_y = (min_y+max_y) / 2.0
    min_dis = 1e10
    min_atom = None
    for atom in all_atoms:
        cur_dis = math.sqrt(math.pow(anchor_x - atom.pos_x, 2) + math.pow(anchor_y - atom.pos_y, 2))
        cur_dis = atom.pos_x * 10000 + (-atom.pos_y)
        if cur_dis < min_dis:
            min_dis = cur_dis
            min_atom = atom
        atom.all_bonds = []
        for bond in atom.out_bonds:
            atom.all_bonds.append((bond, bond.m_angle))
        for bond in atom.in_bonds:
            atom.all_bonds.append((bond, (bond.m_angle - 180) % 360))
        atom.all_bonds = sorted(atom.all_bonds, key=lambda x: x[1])
    start_atom = min_atom


    visited = []
    out_seq = []
    info_arr = []
    branch_arr = []
    ring_branch_arr = []
    dfs_visited(start_atom, None, 0, visited, out_seq, info_arr, withAngle, branch_de_amb=branch_de_amb, branch_arr=branch_arr, bond_dict=bond_dict)

    info_arr = set(info_arr)
    # new_out_text = []
    new_out_seq = []
    new_branch_arr = []
    # new_ring_branch_arr = []
    for ind, element in enumerate(out_seq):
        if ind in info_arr:
            continue  #remove rebundant bracket
        # new_out_text.append(text)
        new_out_seq.append(element)
        new_branch_arr.append(branch_arr[ind])
        # 修改对应的bond_dict索引
        for index, value in enumerate(bond_dict):
            if len(value) > 1 and value[1] == ind:
                bond_dict[index][1] = len(new_out_seq) - 1 # 更新index
                break
        # new_ring_branch_arr.append(ring_branch_arr[ind])
    out_seq = new_out_seq
    branch_arr = new_branch_arr

    #adjust hooks
    cur_uni = 97
    rep_dict = {}
    for element in new_out_seq:
        if not isinstance(element, Atom):
            continue
        if len(element.start_hooks) <= 0:
            continue
        assert len(element.start_hooks) == 1
        hook_name = element.start_hooks[0].m_hookname
        rep_dict[hook_name] = chr(cur_uni)
        cur_uni += 1
    for element in new_out_seq:
        if not isinstance(element, Atom):
            continue
        for hook in element.start_hooks + element.end_hooks:
            hook.m_hookname = rep_dict[hook.m_hookname]

    out_text = []
    out_branch_arr = [] # 键与环相连的相对位置信息
    out_ring_branch_arr = [] # 环上存储键相连的信息
    branch_conn = []
    
    new_bond_dict = []
    for ind, element in enumerate(new_out_seq):
        if isinstance(element, str):
            out_text.append(element)
            out_branch_arr.append(branch_arr[ind])
            out_ring_branch_arr.append(None)
            if len(branch_conn) > 0:
                # if '(' in element:
                #     branch_conn_skip_stack.append('(')
                # elif ')' in element and '(' in branch_conn_skip_stack[-1]: # 匹配到括号
                #     branch_conn_skip_stack.pop(-1)
                # elif len(branch_conn_skip_stack) == 0: # 没有分支
                #     out_ring_branch_arr[-1] = branch_conn.pop(0)
                #     # branch_conn = []
                #     branch_conn_skip_stack = [] # empty stack

                if '(' in element or ')' in element or '-:' in element:
                    pass
                else:
                    out_ring_branch_arr[-1] = branch_conn.pop(0)
            
            # 更新bond_dict
            if '[:' in element and element.endswith(']'): # -[:0]
                for index, value in enumerate(bond_dict):
                    if len(value) > 1 and value[1] == ind:
                        new_bond_dict.append([value[0], len(out_text)-1])
                        break
            # for tmp_index, tmp_value in enumerate(bond_dict):
                

        elif isinstance(element, Atom):
            if isinstance(element, CircleAtom):
                out_text += ["\\circle"]
                out_branch_arr += [None]
                out_ring_branch_arr += [None]
            else:
                out_text += element.normed_text()
                out_branch_arr += [None] * len(element.normed_text())
                out_ring_branch_arr += [None] * len(element.normed_text())
                
                if len(element.ring_branch_info) > 0:
                    branch_conn.append(element.ring_branch_info) # once meet atom, save its ring_branch_info and use it next time

                # 将回连结束标识 ?[a,{-}]也视为Bond
                if len(out_text) > 0 and out_text[-1].startswith('?[') and out_text[-1].endswith(']') and ',' in out_text[-1]:
                    if len(branch_conn) > 0:
                        out_ring_branch_arr[-1] = branch_conn.pop(0)
                    # 从bond_dict中search Bond
                    for bond in bond_dict:
                        if len(bond) == 1 and (element in [bond[0].begin_atom, bond[0].end_atom]):
                            new_bond_dict.append([bond[0], len(out_text)-1])
                
            element.start_hooks = [] # must clear hooks here, other wise it will affect following operation!!!
            element.end_hooks = [] # 
        else:
            raise ValueError
    if len(branch_conn) > 0:
        out_ring_branch_arr[-1] = branch_conn.pop(0)
        assert len(branch_conn) == 0, "还有ring_branch_info未处理."

    assert len(out_branch_arr) == len(out_text), "out_branch_arr必须与out_text长度相同"
    assert len(out_ring_branch_arr) == len(out_text), "out_ring_branch_arr必须与out_text长度相同"
    bond_dict = new_bond_dict
    atom_count = 0
    bond_count = 0
    other_count = 0
    for ele in visited:
        if isinstance(ele, Atom):
            atom_count += 1
        elif isinstance(ele, Bond):
            bond_count += 1
        else:
            other_count += 1
    
    return out_text, out_branch_arr, out_ring_branch_arr, bond_dict

def bond_dict_in_item(item, bond_types):
    '''判断item中是否包含bond_types'''
    for bond in bond_types:
        if bond in item:
            return True
    return False



def split_chemfig(input_chemfig):
    '''将chemfig中相连的单元分开,符合SSML的语法'''
    item_list = input_chemfig.split(' ')
    new_item_list = []

    # 先拆分?[a], ?[a,{-}]合并项
    for id, item in enumerate(item_list):
        if item == '':
            continue
        if '?[' in item and ']' in item: 
            begin_index = item.index('?[')
            # 找到与begin_index匹配的]
            end_index = item[begin_index:].index(']') + begin_index
            
            pre_str = item[: begin_index]
            conn_str = item[begin_index: end_index+1]
            last_str = item[end_index+1: ]
            if pre_str:
                new_item_list.append(pre_str)
            new_item_list.append(conn_str)
            if last_str:
                new_item_list.append(last_str)
        else:
            new_item_list.append(item)
    item_list = new_item_list
    new_item_list = []

    for id, item in enumerate(item_list):
        if item == '':
            continue
        if '[' in item and ']' in item: # 对于化学键,将其之后的分开, eg -[:190]?[a], -[:190]O, ?[a,{-}]?[b,{-}]
            # -[:10]?[a,{-}]?[b,{-}], 多个嵌套
            # N?[a,{-}]
            bond_str = item[: item.index(']')+1]
            last_str = item[item.index(']')+1: ]
            while last_str:
                item = last_str
                new_item_list.append(bond_str)
                if ']' in item:
                    bond_str = item[: item.index(']')+1]
                    last_str = item[item.index(']')+1: ]
                else:
                    bond_str = item
                    last_str = ""
            new_item_list.append(bond_str)
        # 拆开直接的bond 和atom eg, <:H
        elif bond_dict_in_item(item, bond_types):
            for bond in bond_types:
                if bond in item:
                    bond_str = bond
                    last_str = item.replace(bond, '') # 从item中删除Bond
                    new_item_list.append(bond_str)
                    new_item_list.append(last_str)
                    break
        else:
            new_item_list.append(item)
        
    input_chemfig = " ".join(new_item_list)
    return input_chemfig

def show_atom_list(atom_list):
    '''输出原子序列, 只输出原子名'''
    atom_list_name = [atom.name for atom in atom_list]
    atom_name_str = " ".join(atom_list_name)
    print(atom_name_str)

def get_ring_central(atom_list):
    '''获取环所有原子的中心位置'''
    atom_num = len(atom_list)
    total_x = 0
    total_y = 0
    for atom in atom_list:
        total_x += atom.pos_x
        total_y += atom.pos_y
    central_x = total_x / atom_num
    central_y = total_y / atom_num
    return [central_x, central_y]

def get_bond_central(bond):
    '''获取bond的中心点坐标'''
    central_x = (bond.begin_atom.pos_x + bond.end_atom.pos_x) / 2
    central_y = (bond.begin_atom.pos_y + bond.end_atom.pos_y) / 2
    return [central_x, central_y]

def remove_self_ring(all_atoms):
    '''移除分子中从a指向a的键'''
    for atom in all_atoms:
        for bond in atom.in_bonds:
            if bond.begin_atom == bond.end_atom:
                atom.in_bonds.remove(bond)
        for bond in atom.out_bonds:
            if bond.begin_atom == bond.end_atom:
                atom.out_bonds.remove(bond)
    return all_atoms

def reverse_convert_chinese_to_uc(text):
    words = text.split()
    output = []
    for word in words:
        if('\\UC_' in word):
            word = word[-4:]
            word_dec = int(word, 16)
            word = chr(word_dec)
            # print(word)
        if('\\LUC_' in word):
            word = word[-8:]
            word_dec = int(word, 16)
            word = chr(word_dec)
        output.append(word)
    return ' '.join(output)

