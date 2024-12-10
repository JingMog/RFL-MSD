import os, sys
import cv2
from chemfig_struct import *
from chemfig_ops import *
import image_render
import chemfig_parser
import math
import numpy
import pdb
from Tool_Formula.latex_norm.transcription import parse_transcription
import Tool_Formula.latex_norm.transcription as transcription
import argparse
import utils

score_dict = {
    "C": 4,
    "N": 3,
    "O": 2,
    "H": 1
}

def split_words(in_text):
    # in_text ="CH_{3}"
    length = len(in_text)
    ind = 0
    out_arr = [] #[main, info]
    while ind < length:
        ch = in_text[ind]
        if ch in ["_", "^"]:
            out_arr[-1][1] = ch
            ind2 = ind + 1
            while in_text[ind2] != "{":
                if in_text[ind2] != " ":
                    out_arr[-1][1] = in_text[ind2]
                    break
                ind2 += 1
            if in_text[ind2] == "{":
                ind2 += 1
                level = 1
                while ind2 < length:
                    if in_text[ind2] == "{":
                        level += 1
                    elif in_text[ind2] == "}":
                        level -= 1
                    out_arr[-1][1] += in_text[ind2]
                    if level == 0:
                        break
                    ind2 += 1
            ind = ind2 + 1
        elif ch.find("hemabove")!= -1 or ch.find("hembelow")!=-1:
            ind2 = ind + 1
            brakcet_pair_count = 0
            level = 0
            out_arr.append([ch, ""])
            while ind2 < length:
                old_level = level
                if in_text[ind2] == "{":
                    level += 1
                elif in_text[ind2] == "}":
                    level -= 1
                out_arr[-1][1] += in_text[ind2]
                if old_level > 0 and level == 0:
                    brakcet_pair_count += 1
                if brakcet_pair_count >= 2:
                    break
                ind2 += 1
            ind = ind2 + 1
        else:
            out_arr.append([ch, None])
        ind += 1
    return out_arr
    # pdb.set_trace()


def JudgeValidCircle(visited_stack, tgtA, tgtB, used, all_atoms, ring_id=0):
    # pdb.set_trace()
    length = len(visited_stack)
    all_angles = []
    index = length - 1
    success = False
    ring_atoms = []
    while index >= 0:
        curA, curB, _ = visited_stack[index]
        atomA = all_atoms[curA]
        atomB = all_atoms[curB]
        angle = math.atan2(-atomB.pos_y + atomA.pos_y, atomB.pos_x - atomA.pos_x) * 180.0 / math.pi
        all_angles.insert(0, angle)
        ring_atoms.append(atomA)
        if curA == tgtA and curB == tgtB:
            success = True
            break
        index -= 1
    if success is False:
        return None
    edge_num = length - index
    delta_angles = numpy.zeros((edge_num,), dtype=float)
    for i in range(edge_num):
        try:
            cur_delta_angle = all_angles[ (i+1)%edge_num] - all_angles[i%edge_num] + 180
        except:
            pdb.set_trace()
        cur_delta_angle = cur_delta_angle % 180
        delta_angles[i] = cur_delta_angle
    mean = delta_angles.mean()
    mean_var = np.abs(delta_angles - mean).mean()
    # print("mean var = {}".format(mean_var))
    if mean_var >  30:
        return None

    #mark cirlce rings
    atom_num = len(ring_atoms)
    atom_id = 0
    bond_id = 0
    ring_out_bonds = []
    for atom_id, atom in enumerate(ring_atoms):
        atom.__dict__["ring_ids"][ring_id] = (atom_num, atom_id, "*") # should add a "*"
        next_bond = None
        for out_bond in atom.out_bonds:
            if out_bond.end_atom == ring_atoms[(atom_id+1)%atom_num]:
                out_bond.__dict__["ring_ids"][ring_id] = (atom_num, bond_id, "*")
                next_bond = out_bond
        for in_bond in atom.in_bonds:
            if in_bond.begin_atom == ring_atoms[(atom_id+1)%atom_num]:
                in_bond.__dict__["ring_ids"][ring_id] = (atom_num, bond_id, "*")
                next_bond = in_bond
        ring_out_bonds.append(next_bond)
        bond_id += 1
    
    for atom_id, atom in enumerate(ring_atoms):
        if atom.m_text == "":
            continue
        out_trans, bad_trans = reverse_transcription(atom.m_text)
        atom_words = out_trans.split(" ")
        if len(atom_words) <= 0:
            continue
        word_spts = split_words(atom_words)
        if len(word_spts) <= 1:
            continue

        pre_atom = ring_atoms[(atom_id-1)%atom_num]
        next_atom = ring_atoms[(atom_id+1)%atom_num]
        pre_bond = ring_out_bonds[(atom_id-1)%atom_num]
        next_bond = ring_out_bonds[atom_id]
        if (pre_atom.pos_x - atom.pos_x)*(next_atom.pos_x - atom.pos_x) <= 0:
            tgt_pos = 0
            delta = math.fabs(pre_atom.pos_x - atom.pos_x) - math.fabs(next_atom.pos_x - atom.pos_x)
            if math.fabs(delta) < 0.01:
                tgt_pos = 0
                word_inds = [(x, ind + 1) for ind, x in enumerate(word_spts)]
                word_inds = sorted(word_inds, key=lambda x:score_dict[x[0][0]] if x[0][0] in score_dict else 1, reverse=True)
                tgt_pos = word_inds[0][1]
            elif delta > 0:
                tgt_pos = 1
            elif delta < 0:
                tgt_pos = len(word_spts)
            
            if pre_bond.begin_atom == pre_atom:
                pre_bond.m_end = tgt_pos
            else:
                pre_bond.m_start = tgt_pos
            if next_bond.begin_atom == next_atom:
                next_bond.m_end = tgt_pos
            else:
                next_bond.m_start = tgt_pos
            # transcription.split_to_words()       
            # transcription.parse_tree()

    return ring_atoms

def UpdateGraph(ring_atoms, mat, remain_bonds, atom_dict, used):
    atom_num = len(ring_atoms)
    all_node_num = mat.shape[0]
    mask = numpy.ones((all_node_num, all_node_num), dtype=float)
    mask_circle = numpy.zeros((all_node_num,), dtype=float)
    for atom_id, atom in enumerate(ring_atoms):
        curB = atom_dict[atom.name]
        mask_circle[curB] = 1
        curA = atom_dict[ring_atoms[(atom_id+1)%atom_num].name]
        if used[curA, curB] == 0 or used[curB, curA] == 0:
            continue
        mask[curB, curA] = 0
        mask[curA, curB] = 0
     # remove single nodes
    # pdb.set_trace()
    atom_dict_reverse = dict([(v,k) for k, v in atom_dict.items()])
    # pdb.set_trace()
    while True:
        # single_node_ids = ( ((mat*mask).sum(0)<=0) * mask_circle).nonzero()[0].tolist()
        # single_node_ids +=  ( ((mat*mask).sum(0)==0) * (1-mask_circle)).nonzero()[0].tolist()
        single_node_ids = (((mat*mask).sum(0)==0)*mask_circle).nonzero()[0].tolist()
        if len(single_node_ids) <= 0:
            break 
        for single_node_id in single_node_ids:
            #print("remove {}".format(atom_dict_reverse[single_node_id]))
            mat[single_node_id, :] = 0
            mat[:, single_node_id] = 0
            remain_bonds[single_node_id, :] = 0
            remain_bonds[:, single_node_id] = 0
            mask_circle[single_node_id] = 0
    while True:
        single_node_ids = (mat.sum(0)==1).nonzero()[0].tolist()
        if len(single_node_ids) <= 0:
            break 
        for single_node_id in single_node_ids:
            #print("remove {}".format(atom_dict_reverse[single_node_id]))
            mat[single_node_id, :] = 0
            mat[:, single_node_id] = 0
            remain_bonds[single_node_id, :] = 0
            remain_bonds[:, single_node_id] = 0
    pdb.set_trace()

def UpdateGraphGroup(ring_groups, mat, remain_bonds, atom_dict, used):
    group_num = len(ring_groups)
    all_node_num = mat.shape[0]
    mask = numpy.ones((all_node_num, all_node_num), dtype=float)
    mask_circle = numpy.zeros((all_node_num,), dtype=float)
    for ring_atoms in ring_groups:
        atom_num = len(ring_atoms)
        for atom_id, atom in enumerate(ring_atoms):
            curB = atom_dict[atom.name]
            mask_circle[curB] = 1
            curA = atom_dict[ring_atoms[(atom_id+1)%atom_num].name]
            # if used[curA, curB] == 0 or used[curB, curA] == 0:
            #     continue
            mask[curB, curA] = 0
            mask[curA, curB] = 0
     # remove single nodes
    # pdb.set_trace()
    atom_dict_reverse = dict([(v,k) for k, v in atom_dict.items()])
    while True:
        # single_node_ids = ( ((mat*mask).sum(0)<=0) * mask_circle).nonzero()[0].tolist()
        # single_node_ids +=  ( ((mat*mask).sum(0)==0) * (1-mask_circle)).nonzero()[0].tolist()
        cur_delete_num = 0
        single_node_ids = (((mat*mask).sum(0)==0)*mask_circle).nonzero()[0].tolist()
        cur_delete_num += len(single_node_ids)
        for single_node_id in single_node_ids:
            #print("remove {}".format(atom_dict_reverse[single_node_id]))
            mat[single_node_id, :] = 0
            mat[:, single_node_id] = 0
            remain_bonds[single_node_id, :] = 0
            remain_bonds[:, single_node_id] = 0
            mask_circle[single_node_id] = 0
        #pdb.set_trace()
        single_node_ids = (mat.sum(0)==1).nonzero()[0].tolist()
        cur_delete_num += len(single_node_ids)
        for single_node_id in single_node_ids:
            #print("remove {}".format(atom_dict_reverse[single_node_id]))
            mat[single_node_id, :] = 0
            mat[:, single_node_id] = 0
            remain_bonds[single_node_id, :] = 0
            remain_bonds[:, single_node_id] = 0
            mask_circle[single_node_id] = 0
        if cur_delete_num <= 0:
            break
    pass
    # pdb.set_trace()

def backward_stack(visited_stack, used=None):
    nextA = None
    nextB = None
    while len(visited_stack):
        if len(visited_stack[-1][2]) == 0:
            _A, _B, _ = visited_stack.pop()
            if used is not None:
                used[_A, _B] = 0 
            continue
        else:
            next_bond_info = visited_stack[-1][2].pop(0)
            nextB = next_bond_info[3]
            nextA = visited_stack[-1][1]
            break
    return nextA, nextB
    
def stack_find(visited_stack, A, B):
    length = len(visited_stack)
    for i in range(length-1, -1, -1):
        if visited_stack[i][0] == A and visited_stack[i][1] == B:
            return True
    return False

def FindRings(in_atom:Atom, start_ring_id=0, circle_ring_groups=[]):
    all_atoms = GetAllAtoms(in_atom)
    #pdb.set_trace()
    atom_dict = dict([ (atom.name, ind) for ind, atom in enumerate(all_atoms) ])
    atom_num = len(all_atoms)
    used = np.zeros((atom_num, atom_num), dtype=numpy.int32) # is side used

    mat = np.zeros((atom_num, atom_num), dtype=numpy.int32) # adj_mat
    remain_bonds = np.zeros((atom_num, atom_num), dtype=Bond) # bonds that are not deleted
    # remain_bonds = {} # (atom1, atom2, bond)
    for atom in all_atoms:
        for bond in atom.in_bonds + atom.out_bonds:
            atom_id1 = atom_dict[bond.begin_atom.name]
            atom_id2 = atom_dict[bond.end_atom.name]
            mat[atom_id1, atom_id2] = 1
            mat[atom_id2, atom_id1] = 1
            remain_bonds[atom_id1, atom_id2] = bond
            remain_bonds[atom_id2, atom_id1] = bond
    # remove single nodes
    # while True:
    #     single_node_ids = (mat.sum(0)==1).nonzero()[0].tolist()
    #     if len(single_node_ids) <= 0:
    #         break 
    #     for single_node_id in single_node_ids:
    #         mat[single_node_id, :] = 0
    #         mat[:, single_node_id] = 0
    #         remain_bonds[single_node_id, :] = 0
    #         remain_bonds[:, single_node_id] = 0
    # pdb.set_trace()
    UpdateGraphGroup(circle_ring_groups, mat, remain_bonds, atom_dict, used)
    # idxs = remain_bonds.nonzero()[0].tolist()
    # atom_list = set([all_atoms[x].name for x in idxs])

    # dfs
    #pdb.set_trace()
    ring_id = start_ring_id
    all_circles = []
    all_ring_groups = {}
    while True:
        cand_pair = [(a,b) for a, b in zip(*remain_bonds.nonzero())]
        if len(cand_pair) <=0 :
            break
        ring_groups = []
        #print("A-------------------------")
        while len(cand_pair) > 0:
            curA, curB = cand_pair[-1]
            visited_stack = [] # (curA, curB, branches[] )
            find = False
            #print("B-------------------------")
            while True:
                atomA = all_atoms[curA]
                atomB = all_atoms[curB]
                #print("visited atomA={} atomB={}".format(atomA.name, atomB.name))
                # if atomA.name == "Atom_9" and atomB.name=="Atom_1":
                #     pdb.set_trace()
                # if curA == 6 and curB == 0:
                #     pdb.set_trace()
                # pdb.set_trace()
                used[curA, curB] = 1
                if (curA, curB) in cand_pair:
                    cand_pair.remove((curA, curB))
                angle = math.atan2(-atomB.pos_y + atomA.pos_y, atomB.pos_x - atomA.pos_x) * 180.0 / math.pi
                all_next_bonds = []
                for _bond, _angle in atomB.all_bonds:
                    _tgt_atom = _bond.begin_atom if atomB == _bond.end_atom else _bond.end_atom
                    if _tgt_atom == atomA:
                        continue
                    _nextC = atom_dict[_tgt_atom.name]
                    if mat[curB, _nextC] == 0:
                        continue
                    is_cand = stack_find(visited_stack, curB, _nextC)
                    if used[curB, _nextC] > 0 and not is_cand:
                        continue
                    #pdb.set_trace()
                    delta_angle = (_angle - (angle + 180) ) % 360
                    if delta_angle > 180:
                        continue
                    all_next_bonds.append((_bond, _angle, _tgt_atom, _nextC, is_cand, delta_angle)) #right most            
                all_next_bonds = sorted(all_next_bonds, key = lambda x:x[-1])
                visited_stack.append((curA, curB, all_next_bonds[1:]))
                nextA = None
                nextB = None
                if len(all_next_bonds) == 0:
                    nextA, nextB = backward_stack(visited_stack, None)
                else:
                    nextB = all_next_bonds[0][3]
                    nextA = curB
                    # if used[nextA, nextB] == 1: #already used, we find circle cand
                    # if stack_find(visited_stack, nextA, nextB):
                    if all_next_bonds[0][4] > 0:
                        ring_atoms = JudgeValidCircle(visited_stack, nextA, nextB, used, all_atoms, ring_id)
                        if ring_atoms is not None:
                            ring_groups.append(ring_atoms)
                            all_ring_groups[ring_id] = ring_atoms
                            #pdb.set_trace()
                            # UpdateGraph(ring_atoms, mat, remain_bonds, atom_dict, used)
                            find = True
                            all_circles.append([a.name for a in ring_atoms])
                            ring_id += 1
                            # break
                            nextA, nextB = backward_stack(visited_stack)
                        else:
                            nextA, nextB = backward_stack(visited_stack, used)
                if nextA is None or nextB is None:
                    break
                curA = nextA
                curB = nextB
        
        UpdateGraphGroup(ring_groups, mat, remain_bonds, atom_dict, used)
        #all_ring_groups += ring_groups
        #print("delete end")
        #pdb.set_trace()
        if not find:
            break
        # pdb.set_trace()
    #pdb.set_trace()
    # print("====================")
    #pdb.set_trace()
    return all_ring_groups

def AdjustRings(circle_ring_groups):
    all_ring_bonds = {} #{ring_id: [bond]}
    is_assigned = {} #[bond_name, 0 or 1]
    begin_atom = None
    # pdb.set_trace()
    for group_id, ring_atoms in circle_ring_groups.items():
        ring_bonds = []
        for atom_id, atom in enumerate(ring_atoms):
            tgt_bond = None
            for bond in atom.out_bonds:
                if bond.end_atom == ring_atoms[(atom_id+1)%len(ring_atoms)]:
                    ring_bonds.append(bond)
                    tgt_bond = bond
            if tgt_bond is None:
                for bond in atom.in_bonds:
                    if bond.begin_atom == ring_atoms[(atom_id+1)%len(ring_atoms)]:
                        ring_bonds.append(bond)
                        tgt_bond = bond
            if begin_atom is None:
                begin_atom = atom
            # if tgt_bond is None:
            #     pdb.set_trace()
            is_assigned[tgt_bond.name] = 0
        all_ring_bonds[group_id] = ring_bonds
    remain_ring_ids = list(circle_ring_groups.keys())
    while len(remain_ring_ids) > 0:
        max_ring_id = -1
        max_score = -10000
        for ring_id in remain_ring_ids:
            all_bonds = all_ring_bonds[ring_id]
            assign_num = 0
            shared_num = 0
            for bond in all_bonds:
                if is_assigned[bond.name] > 0:
                    assign_num += 1
                elif len(bond.ring_ids) > 1:
                    shared_num += 1
            score = assign_num*100 + shared_num
            if score > max_score:
                max_score = score
                max_ring_id = ring_id
        remain_ring_ids.remove(max_ring_id)

        ref_id = -1
        ref_angle = None
        ref_length = None
        ring_bonds = all_ring_bonds[max_ring_id]
        ring_atoms = circle_ring_groups[max_ring_id]
        edge_num = len(ring_bonds)
        angle_step = 360.0/edge_num
        corrupt = False
        for bid, bond in enumerate(ring_bonds):
            need_reverse = (bond.begin_atom == ring_atoms[(bid+1)%edge_num])
            if is_assigned[bond.name] > 0:
                if ref_id == -1:
                    ref_angle = bond.m_angle
                    if need_reverse:
                        ref_angle = (ref_angle+180)%360
                    ref_length = bond.m_length
                    ref_id = bid
                else: # check corrupt
                    cur_angle = bond.m_angle
                    if need_reverse:
                        cur_angle = (cur_angle+180)%360
                    delta_angle = cur_angle - ref_angle
                    if math.fabs(delta_angle -  (bid - ref_id)*angle_step) > 1e-4:
                        # corrupt, delete it!
                        for bond in ring_bonds:
                            if max_ring_id in bond.ring_ids:
                                bond.ring_ids.pop(max_ring_id)
                            else:
                                pass
                                # warnings.warn("delete ring err! can not find ring_id {} in {}".format(max_ring_id, bond))
                            corrupt = True
                    else:
                        pass
        if corrupt:
            continue
        if ref_id == -1:
            hist = {}
            length_step = 0.2
            for bid, bond in enumerate(ring_bonds):
                key = round(bond.m_length / length_step)
                if key not in hist:
                    hist[key] = [bid]
                else:
                    hist[key].append(bid)
            max_key=-1
            max_count =-1
            for key in hist:
                if len(hist[key]) > max_count:
                    max_count = len(hist[key])
                    max_key = key
            ref_id = hist[max_key][0]
            ref_length = ring_bonds[ref_id].m_length
            ref_angle = ring_bonds[ref_id].m_angle
            need_reverse = (ring_bonds[ref_id].begin_atom == ring_atoms[(ref_id+1)%edge_num])
            if need_reverse:
                ref_angle = (ref_angle+180)%360
        # assign angle
        
        for bid, bond in enumerate(ring_bonds):
            if is_assigned[bond.name] == 0:
                need_reverse = (bond.begin_atom == ring_atoms[(bid+1)%edge_num])
                desired_angle = ref_angle + (bid - ref_id) * angle_step
                if need_reverse:
                    desired_angle = ( desired_angle + 180 )% 360
                bond.m_angle = desired_angle
                bond.m_length = ref_length
                is_assigned[bond.name] = 1
    
    all_atoms = SimulateCoord(begin_atom, scale=1)
    for atom in all_atoms:
        atom.all_bonds = []
        for bond in atom.out_bonds:
            atom.all_bonds.append((bond, bond.m_angle))
        for bond in atom.in_bonds:
            atom.all_bonds.append((bond, (bond.m_angle - 180) % 360))
        atom.all_bonds = sorted(atom.all_bonds, key=lambda x: x[1])
    # SimulateCoord(begin_atom, scale = 100)
    # cv2.imwrite("debug/adjust.jpg", image_render.rend_atoms(GetAllAtoms(begin_atom), scale=100))
    # SimulateCoord(begin_atom, scale=1)
    # pdb.set_trace()
    return
        
            





        
        
        
        
            
            

# reverse rend
def MarkCircleRing(circleAtom: Atom, ring_id=0, base=10000):
    # assert len(circleAtom.out_bonds) +  len(circleAtom.in_bonds) == 1
    if len(circleAtom.out_bonds) +  len(circleAtom.in_bonds) != 1:
        return 0, []
    # pdb.set_trace()
    if len(circleAtom.out_bonds) == 1: #circle -> begin
        begin_atom = circleAtom.out_bonds[0].end_atom
        origin_length = circleAtom.out_bonds[0].m_length
        begin_atom.in_bonds.remove(circleAtom.out_bonds[0])
        circleAtom.out_bonds.clear()
    else: #circle <- begin
        begin_atom = circleAtom.in_bonds[0].begin_atom
        origin_length = circleAtom.in_bonds[0].m_length
        begin_atom.out_bonds.remove(circleAtom.in_bonds[0])
        circleAtom.in_bonds.clear()
    new_all_bonds = []
    for info in begin_atom.all_bonds:
        if info[0].begin_atom == circleAtom or info[0].end_atom == circleAtom :
            continue
        new_all_bonds.append(info)
    begin_atom.all_bonds = new_all_bonds

    # begin_atom = (circleAtom.in_bonds + circleAtom.out_bonds)[0].begin_atom
    # if begin_atom == circleAtom:
    #     begin_atom = (circleAtom.in_bonds + circleAtom.out_bonds)[0].end_atom
    ref_angle = math.atan2(-begin_atom.pos_y + circleAtom.pos_y, begin_atom.pos_x - circleAtom.pos_x) * 180.0 / math.pi
    ref_length = math.sqrt(math.pow(begin_atom.pos_y - circleAtom.pos_y, 2) + math.pow(begin_atom.pos_x - circleAtom.pos_x, 2))
    visited = set()
    path = []
    
    def dfs_find(curAtom, lastBond=None, visited=[]):
        if curAtom in visited:
            return [visited]
        cur_visited = visited + [curAtom]
        cur_angle = math.atan2(-curAtom.pos_y + circleAtom.pos_y, curAtom.pos_x - circleAtom.pos_x) * 180.0 / math.pi
        ret = []
        for bond in curAtom.out_bonds + curAtom.in_bonds:
            next_atom = bond.end_atom if bond.end_atom != curAtom else bond.begin_atom
            if len(visited)>0 and next_atom == visited[-1]:
                continue
            next_angle = math.atan2(-next_atom.pos_y + circleAtom.pos_y, next_atom.pos_x - circleAtom.pos_x) * 180.0 / math.pi
            next_length = math.sqrt(math.pow(next_atom.pos_y - circleAtom.pos_y, 2) + math.pow(next_atom.pos_x - circleAtom.pos_x, 2))
            # if (next_angle-cur_angle) % 180 >= 90:
            #     continue
            if (next_angle-cur_angle) % 360 >= 180:
                continue
            if math.fabs(next_length - ref_length) > ref_length * 0.5:
                continue
            ret += dfs_find(next_atom, bond, cur_visited)
        return ret
    det_results = dfs_find(begin_atom, None, [])
    if len(det_results) <= 0:
        det_results = [[begin_atom]]
    # assert len(det_results) == 1
    ring_atoms = det_results[0]
    # ring_atoms = sorted(ring_atoms, key = lambda x:x.pos_x*base - x.pos_y)

    atom_num = len(ring_atoms)
    atom_id = 0
    bond_id = 0
    # pdb.set_trace()
    if atom_num > 1:
        for atom_id, atom in enumerate(ring_atoms):
            # if "ring_ids" not in atom.__dict__:
            #     atom.__dict__["ring_ids"] = {}
            # atom.__dict__["ring_ids"].append((ring_id, atom_num, atom_id))
            atom.__dict__["ring_ids"][ring_id] = (atom_num, atom_id, "**")
            for out_bond in atom.out_bonds:
                if out_bond.end_atom == ring_atoms[(atom_id+1)%atom_num]:
                    # if "ring_ids" not in out_bond.__dict__:
                    #     out_bond.__dict__["ring_ids"] = {}
                    # out_bond.__dict__["ring_ids"].append((ring_id, atom_num))
                    out_bond.__dict__["ring_ids"][ring_id] = (atom_num, bond_id, "**")
                    #print(out_bond)
            for in_bond in atom.in_bonds:
                if in_bond.begin_atom == ring_atoms[(atom_id+1)%atom_num]:
                    # if "ring_ids" not in in_bond.__dict__:
                    #     in_bond.__dict__["ring_ids"] = {}
                    # in_bond.__dict__["ring_ids"].append((ring_id, atom_num))
                    in_bond.__dict__["ring_ids"][ring_id] = (atom_num, bond_id, "**")
                    #print(in_bond)
            bond_id += 1
    else:
        atom_num = 0
        ring_atoms = []
    return atom_num, ring_atoms

def JudgeEncounterParentRing(cur_atom, ring_stack, last_ring_id):
    if last_ring_id == -1:
        cur_ring_stack = ring_stack + [-1]
    for pre_ring_id in ring_stack:
        if pre_ring_id == last_ring_id:
            break
        if pre_ring_id in cur_atom.ring_ids:
            return True
    return False

def CalcDefaultAngle(cur_atom, last_bond, last_angle, last_ring_id, new_edge_num=6, old_edge_num =6, child_id = 0):
    # pdb.set_trace()
    ring_last_angle = None
    inner_angle = 180.0*(new_edge_num-2.0)/new_edge_num
    # calc absoluted zero angle
    if last_bond is None and child_id == 0: # logic of "is_first"
        ring_last_angle = 90
        last_angle = 0
    else:
        ring_last_angle = inner_angle/2.0
    if last_ring_id == -1: # not in ring
        # defaultAngle = ring_last_angle - inner_angle
        # defaultAngle = defaultAngle % 360
        absZeroAngle = (ring_last_angle - inner_angle)%360
        defaultAngle = (last_angle - inner_angle/2)%360
    else:
        # default angle
        ring_last_angle = None
        for bond, angle in cur_atom.all_bonds:
            if bond != last_bond and last_ring_id in bond.ring_ids:
                ring_last_angle = angle
                break
        defaultAngle = ring_last_angle - inner_angle
        defaultAngle = defaultAngle % 360
        # abs zero angle
        absZeroAngle = 0
        parent_inner_angle = 180 * (old_edge_num-2) / old_edge_num
        ring_last_angle = (180 - parent_inner_angle)%360
        absZeroAngle = (ring_last_angle - inner_angle)%360
    return absZeroAngle, defaultAngle

def dfs_visited_for_reverse_rend(cur_atom, last_bond=None, last_angle=0, last_ring_id=-1, visited=[], out_text=[], info_arr=[], withAngle=1, ring_stack=[]):
    top_ring = -1 if len(ring_stack) == 0 else ring_stack[-1]
    if last_bond in visited:
        return

    if last_bond is not None:
        if (not cur_atom in visited and not JudgeEncounterParentRing(cur_atom, ring_stack, last_ring_id)) or last_ring_id in cur_atom.ring_ids:
            if last_bond.m_type is not None and last_bond.m_type != "" and (last_bond.m_type != "-:") and (not isinstance(cur_atom, CircleAtom) or (not cur_atom.m_text == "\\circle")):
                # pdb.set_trace()
                if last_bond.m_type != "=:":
                    out_text.append(last_bond.m_type)
                else:
                    out_text.append("-")
                bond_info = ["", "", "", "", ""]
                if withAngle > 0 and len(last_bond.ring_ids)<=0:
                    last_angle = last_angle % 360
                    bond_info[0] = ":{}".format(int(last_angle))
                if math.fabs(last_bond.m_length - 1) > 0.01:
                    length = last_bond.m_length
                    length = round(length*1e6)/float(1e6)
                    bond_info[1] = "{}".format(length)
                if last_bond.end_atom == cur_atom:
                    if last_bond.m_start is not None and last_bond.m_start > 0:
                        bond_info[2] = "{}".format(last_bond.m_start)
                    if last_bond.m_end is not None and last_bond.m_end > 0:
                        bond_info[3] = "{}".format(last_bond.m_end)
                else:
                    if last_bond.m_start is not None and last_bond.m_start > 0:
                        bond_info[3] = "{}".format(last_bond.m_start)
                    if last_bond.m_end is not None and last_bond.m_end > 0:
                        bond_info[2] = "{}".format(last_bond.m_end)
                if last_bond.m_type == "=:":
                    bond_info[4] = "draw=none"
                max_info_id = len(bond_info)
                while bond_info[max_info_id-1] == "" and max_info_id > 0:
                    max_info_id -= 1
                if max_info_id > 0:
                    # out_text[-1] += "[:{}]".format(int(last_angle))
                    out_text[-1] += "[" + ",".join(bond_info[:max_info_id]) + "]"


        else: # create hook
            start_atom = cur_atom
            if cur_atom == last_bond.end_atom:
                end_atom = last_bond.begin_atom
            else:
                end_atom = last_bond.end_atom
            if len(start_atom.start_hooks) == 0:
                hook_uni = 97 #97~122
                while hook_uni in visited:
                    hook_uni += 1
                if hook_uni > 122:
                    raise ValueError("File {}, line {} :not enough hook name".format(__file__, sys._getframe().f_lineno))
                visited.append(hook_uni)
                hook_name = chr(hook_uni)
                new_start_hook = DistantHook("?[{}]".format(hook_name))
                start_atom.start_hooks.append(new_start_hook)
            if len(start_atom.start_hooks) > 0: #already exists
                new_end_hook = DistantHook(start_atom.start_hooks[0].attr_str)
                new_end_hook.m_bondtype = last_bond.m_type
                end_atom.end_hooks.append(new_end_hook)
            pass
        visited.append(last_bond)

    # if cur_atom.name == "Atom_1":
    #     pdb.set_trace()

    if last_ring_id != -1 and cur_atom in visited and last_ring_id in cur_atom.ring_ids:
        # ring_stack.remove(last_ring_id)
        return
    
    if JudgeEncounterParentRing(cur_atom, ring_stack, last_ring_id):
        # if last_ring_id != -1:
        #     ring_stack.remove(last_ring_id)
        return

    if cur_atom in visited:
        return
    #print(cur_atom)
    # print("{}\t{}\t{}\t{}\t{}".format(last_ring_id, cur_atom, last_bond, last_angle, ring_stack))
    # if cur_atom.name == "Atom_5":
    #     pdb.set_trace()

    if cur_atom is not None:
        out_text.append(cur_atom)
        visited.append(cur_atom)
    
    #decide next atom

    all_bonds = []
    for bond, angle in cur_atom.all_bonds:
        if bond == last_bond:
            continue
        if cur_atom == bond.end_atom:
            if len(bond.ring_ids) == 0:
                all_bonds.append((bond, angle, bond.begin_atom, -1))
            else:
                for ring_id in bond.ring_ids:
                    all_bonds.append((bond, angle, bond.begin_atom,ring_id))
        elif cur_atom == bond.begin_atom:
            if len(bond.ring_ids) == 0:
                all_bonds.append((bond, angle, bond.end_atom, -1))
            else:
                for ring_id in bond.ring_ids:
                    all_bonds.append((bond, angle, bond.end_atom, ring_id))
        else:
            raise ValueError("atom bond not connect")

    pairs = []
    last_ind = -1
    
    
    # if last_ring_id == -1: #out ring
    #     pass
    # else: # in ring
    bond_score = {}
    for bid, (bond, angle, next_atom, tgt_ring_id) in enumerate(all_bonds):
        #-----------------
        has_new_circle_ring = 0
        if tgt_ring_id not in ring_stack and tgt_ring_id != -1:
            # judge clock direction
            cur_bond_ring_rank = bond.ring_ids[tgt_ring_id][1]
            edge_num = bond.ring_ids[tgt_ring_id][0]
            adj_bond_ring_rank = -1
            for adj_bond, _, _, _ in all_bonds:
                if adj_bond == bond:
                    continue
                if tgt_ring_id in adj_bond.ring_ids:
                    adj_bond_ring_rank = adj_bond.ring_ids[tgt_ring_id][1]
                    break
            if adj_bond_ring_rank == -1:
                has_new_circle_ring = 0
            else:
                if math.fabs(cur_bond_ring_rank - adj_bond_ring_rank) > 1:
                    if cur_bond_ring_rank < adj_bond_ring_rank:
                        cur_bond_ring_rank += edge_num #
                    else:
                        adj_bond_ring_rank += edge_num
                if cur_bond_ring_rank > adj_bond_ring_rank:
                    has_new_circle_ring = 1
                else:
                    has_new_circle_ring = 0
                    all_bonds[bid] = (bond, angle, next_atom, -1)
                    tgt_ring_id = -1
        #-------------------
        on_last_ring = 0
        if tgt_ring_id == last_ring_id:
            on_last_ring = 1
        #-----------------
        normal_branch = 1 if len(bond.ring_ids) == 0 else 0
        #---------------total score
        bond_score["{}-{}".format(bond.name, tgt_ring_id)] = has_new_circle_ring * 100 + normal_branch * 10 + on_last_ring
    all_bonds = sorted(all_bonds, key = lambda x: bond_score["{}-{}".format(x[0].name, x[-1])], reverse=True )

    # if cur_atom.name == "Atom_1":
    #     pdb.set_trace()

    branch_type = []    
    for child_id, (bond, angle, next_atom, tgt_ring_id) in enumerate(all_bonds):
        added_circle_ring = False
        if tgt_ring_id != last_ring_id and tgt_ring_id != -1:
            if tgt_ring_id not in ring_stack:
                # pdb.set_trace()
                ring_stack.append(tgt_ring_id)
                # pdb.set_trace()
                old_edge_num = -1
                if last_ring_id != -1:
                    old_edge_num = last_bond.ring_ids[last_ring_id][0]
                # pdb.set_trace()
                absZeroAngle, defaultAngle = CalcDefaultAngle(cur_atom, last_bond, last_angle, last_ring_id, bond.ring_ids[tgt_ring_id][0], old_edge_num=old_edge_num, child_id=child_id)
                absAngle = round(angle - absZeroAngle)%360
                # if deltaAngle > 180:
                #     deltaAngle = deltaAngle - 360
                out_text.append("({}{}([:{}]".format(bond.ring_ids[tgt_ring_id][2], bond.ring_ids[tgt_ring_id][0], absAngle))
                # if deltaAngle == 0:
                #     out_text.append("({}{}(".format(bond.ring_ids[tgt_ring_id][2], bond.ring_ids[tgt_ring_id][0]))
                # else:
                #     out_text.append("({}{}([:{}]".format(bond.ring_ids[tgt_ring_id][2], bond.ring_ids[tgt_ring_id][0], deltaAngle))
                begin_ind = len(out_text) - 1
                added_circle_ring = True
                #pdb.set_trace()
        elif tgt_ring_id == last_ring_id and last_ring_id != -1:
            pass
        else:
            out_text.append("(")
            begin_ind = len(out_text) - 1

        dfs_visited_for_reverse_rend(next_atom, bond, angle, tgt_ring_id, visited, out_text, info_arr, withAngle, ring_stack)

        if tgt_ring_id != last_ring_id and tgt_ring_id != -1:
            if added_circle_ring:
                out_text.append("))")
                end_ind = len(out_text) - 1
                pairs.append((begin_ind, end_ind))
                branch_type.append("ring_head")
            else:
                pairs.append((None, None))
                branch_type.append("in_ring")
        elif tgt_ring_id == last_ring_id and last_ring_id != -1:
            pairs.append((None, None))
            branch_type.append("in_ring")
            pass
        else:
            out_text.append(")")
            end_ind = len(out_text) - 1
            pairs.append((begin_ind, end_ind))
            branch_type.append("normal_branch")
            # if end_ind == begin_ind + 1:
            #     info_arr += [begin_ind, end_ind]
            #     last_ind = len(pairs) - 2
            # else:
            #     last_ind = len(pairs) - 1

    length = len(pairs)
    last_ind = -1
    for i in range(length-1, -1, -1):
        begin, end = pairs[i]
        if begin is not None and end is not None and end == begin + 1:
            continue
        last_ind = i
        break        
    if last_ind != -1 and branch_type[last_ind] != "ring_head":
        info_arr += list(pairs[last_ind])
    for begin, end in pairs:
        if begin is not None and end is not None and end == begin + 1:
            info_arr += [begin, end]
    

    # if len(pairs) > 0 and last_ind >= 0 and tgt_ring_id == -1:
    #     info_arr += list(pairs[last_ind])

def pre_process_for_reverse(rootAtom, scale = 1):
    all_atoms = SimulateCoord(rootAtom, scale=scale)
    all_atoms = RemoveDupAtoms(all_atoms[0], th=scale*0.01)
    RemoveDupBonds(all_atoms[0])
    ConnectDistantAtoms(all_atoms[0])
    all_atoms = NormAllCircleAtom(all_atoms[0])
    return all_atoms

def reverse_rend_text(rootAtom, scale=1, remove_dup=1, norm_circle=0, connect_distant=1):
    withAngle = 1
    # cv2.imwrite("debug.jpg", rend(rootAtom,scale=100))
    # cv2.imwrite("debug_name.jpg", rend(rootAtom,scale=100, rend_name=1))
    all_atoms = GetAllAtoms(rootAtom)
    min_x, min_y, max_x, max_y = GetCoordRange(all_atoms)
    
    # pdb.set_trace()

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

    # #adjust atom pos
    # mean_posy = (max_y + min_y) / 2.0
    # delta_length = start_atom.pos_y - mean_posy
    # angle = 90
    # if delta_length > 0:
    #     angle = 90
    # else:
    #     angle = -90
    # length = math.fabs(delta_length)
    
    # anchor_atom = Atom("")
    # anchor_atom.pos_x = start_atom.pos_x
    # anchor_atom.pos_y = mean_posy
    # anchor_bond = Bond("-:")
    # anchor_bond.begin_atom = anchor_atom
    # anchor_bond.end_atom = start_atom
    # anchor_bond.m_length = length
    # start_atom = anchor_atom

    

    #FindRings(start_atom)

    ring_id = 0
    ring_dict = {}
    circle_ring_dict = {}
    circle_ring_groups = []
    for atom in all_atoms:
        if isinstance(atom, CircleAtom) or atom.m_text == "\\circle":
            edge_num, ring_atoms = MarkCircleRing(atom, ring_id=ring_id)
            if edge_num <= 1:
                continue
            ring_dict[ring_id] = edge_num
            circle_ring_dict[ring_id] = ring_atoms
            circle_ring_groups.append(ring_atoms)
            ring_id += 1

    normal_ring_dict = FindRings(start_atom, start_ring_id=ring_id, circle_ring_groups = circle_ring_groups)
    normal_ring_dict.update(circle_ring_dict)
    
    AdjustRings(normal_ring_dict)
    min_x, min_y, max_x, max_y = GetCoordRange(all_atoms)
    # ssss
    #adjust atom pos
    find_charge = False
    for atom in all_atoms:
        if isinstance(atom, Charge):
            find_charge=True
    if not find_charge:
        mean_posy = (max_y + min_y) / 2.0 + 0.2
        #mean_posy += 0.2
        # sum_posy = 0
        # for atom in all_atoms:
        #     sum_posy += atom.pos_y
        # mean_posy = sum_posy/len(all_atoms)
        delta_length = start_atom.pos_y - mean_posy
        angle = 90
        if delta_length > 0:
            angle = -90
        else:
            angle = 90
        length = math.fabs(delta_length)
        length = round(length*1e6)/1e6
        # length = length * 2 / 3.0
        if length > 0.00001:
            anchor_atom = Atom("")
            anchor_atom.pos_x = start_atom.pos_x
            anchor_atom.pos_y = mean_posy
            anchor_bond = Bond("=:")
            anchor_bond.begin_atom = anchor_atom
            anchor_bond.end_atom = start_atom
            anchor_bond.m_length = length
            anchor_atom.out_bonds.append(anchor_bond)
            anchor_atom.all_bonds = []
            anchor_atom.all_bonds.append((anchor_bond, angle))
            start_atom = anchor_atom

    # pdb.set_trace()
    visited = []
    out_seq = []
    info_arr = []
    ring_stack=[]
    dfs_visited_for_reverse_rend(start_atom, None, 0, -1, visited, out_seq, info_arr, withAngle, ring_stack)
    # pdb.set_trace()
    info_arr = set(info_arr)
    #new_out_text = []
    new_out_seq = []
    for ind, element in enumerate(out_seq):
        if ind in info_arr:
            continue  #remove rebundant bracket
        # new_out_text.append(text)
        new_out_seq.append(element)
    out_seq = new_out_seq

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
    for element in new_out_seq:
        if isinstance(element, str):
            out_text.append(element)
        elif isinstance(element, Atom):
            if isinstance(element, CircleAtom):
                out_text += ["\\circle"]
            else:
                # out_text += element.normed_text()
                out_text += element.reverse_normed_text()
        elif isinstance(element, Bond):
            last_bond = element
            cur_atom = last_bond.end_atom
            angle = math.atan2(last_bond.end_atom.pos_y - last_bond.begin_atom.pos_y, last_bond.end_atom.pos_x - last_bond.begin_atom.pos_x)
            if last_bond.m_type is not None and last_bond.m_type != "" and (last_bond.m_type != "-:" or isinstance(cur_atom, CircleAtom) or cur_atom.m_text == "\\circle"):
                out_text.append(last_bond.m_type)
            if withAngle > 0:
                # normed_angle = int((angle+7.5) / 15.0) * 15
                # normed_angle = normed_angle % 360
                out_text.append("[:{}]".format(angle))



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
    # pdb.set_trace()
    # assert atom_count == len(all_atoms), "{} vs {}".format(atom_count, len(all_atoms))

    return out_text


valid_chars = []
eng_chars = [chr(x) for x in range(ord("a"), ord("z"))] +   [chr(x) for x in range(ord("A"), ord("Z"))]
valid_chars += eng_chars
num_chars = [chr(x) for x in range(ord("0"), ord("9"))]
valid_chars += num_chars
special_chars = ["_", "^", "{", "}", "-", "=", "\\equiv"]
valid_chars += special_chars

def JudgeValidFlatChem(cur_text):
    # pdb.set_trace()
    if cur_text.find("-") == -1 and cur_text.find("\\equiv") == -1:
        return False
    if cur_text.count("{") != cur_text.count("}"):
        return False
    find_eng = False
    for ind, ch in enumerate(cur_text):
        if ch == "{":
            if ind == 0:
                return False
            elif cur_text[ind-1] not in ["_", "^"]:
                return False
        if ch in eng_chars:
            find_eng = True
    if not find_eng:
        return False
    
    cur_text_rep = cur_text.replace("\\equiv", "~")
    spts = cur_text.split("-")
    spts += cur_text.split("~")
    
    #spts = [x for x in spts if x != ""]
    for spt in spts:
        find_eng = False
        for ch in spt:
            if ch in eng_chars:
                find_eng = True
                break
        if not find_eng:
            return False
        if spt.count("{") != spt.count("}"):
            return False

    return True
    
def process_flat_chemfig(text, node_scale=1, atom_seq=2):
    cur_begin = None
    cur_end = None
    all_ranges = []
    cur_text = ""
    curLevel = 0
    for ind, ch in enumerate(text):
        if ch in valid_chars:
            if curLevel <= 0:
                if cur_begin is None:
                    cur_begin = ind
                    cur_text = ch
                else:
                    cur_text += ch
            if ch == "{" and (ind ==0 or text[ind-1] not in {"_", "^"}):
                cur_begin = None
                cur_end = None
                cur_text = ""
                curLevel += 1
            elif curLevel > 0:
                if ch == "}":
                    curLevel -= 1
                elif ch == "{":
                    curLevel += 1
        else:
            if cur_begin is not None:
                cur_end = ind
                # pdb.set_trace()
                if JudgeValidFlatChem(cur_text):
                    all_ranges.append((cur_begin, cur_end))
                cur_end = None
                cur_begin = None
                cur_text = ""
        # print("{}-{}-{}".format(ch, cur_begin, curLevel))
    out_text = []
    last_end = 0
    # pdb.set_trace()
    #out_text += text[:all_ranges[0][0]]
    for ind, (begin, end) in enumerate(all_ranges):
        out_text += text[last_end: begin]
        out_text.append("\\chemfig[node style={{scale={}}},atom sep={}em]{{".format(node_scale, atom_seq))
        # out_text += text[begin:end]
        for ch in text[begin:end]:
            if ch == "\\equiv":
                out_text.append("~")
            else:
                out_text.append(ch)
        out_text.append("}")
        last_end = end
        #print("find = {}".format( "".join(text[begin:end])))
    out_text += text[last_end:]
    return out_text

def reverse_chemfig(inputStr):
    begin = inputStr.find("\\chemfig")
    left = inputStr.find("{",begin)
    right = inputStr.rfind("}")
    if left != -1 and right != -1:
        chemfig_str = inputStr[left+1:right]
    else:
        chemfig_str = inputStr
    root_atom, _ = chemfig_parser.chemfig_parse(chemfig_str)
    all_atoms = pre_process_for_reverse(root_atom)
    out_text_arr = reverse_rend_text(all_atoms[0])
    return out_text_arr, all_atoms

# interface for reverse render
def reverse_organic_trans(inputStr, node_scale=1, atom_seq=1.75, debug=False):
    _, rep_dict, text = utils.replace_chemfig(inputStr)
    if text is None:
        return inputStr
    new_text, _ = reverse_transcription(text)

    new_text = new_text.split(" ")
    new_text = process_flat_chemfig(new_text, node_scale, atom_seq)

    out_text = []
    for unit in new_text:
        if unit in rep_dict:
            chemfig_str = rep_dict[unit].strip()[8:].strip()[1:-1]
            root_atom, _ = chemfig_parser.chemfig_parse(chemfig_str)
            all_atoms = pre_process_for_reverse(root_atom)
            if debug:
                all_atoms = SimulateCoord(all_atoms[0], scale=100)
                img_key = unit.replace("\\chem", "")
                cv2.imwrite("./debug/reverse_rend_{}.jpg".format(img_key), image_render.rend_atoms(all_atoms, scale=100, rend_name=0))
                cv2.imwrite("./debug/reverse_rend_name_{}.jpg".format(img_key), image_render.rend_atoms(all_atoms, scale=100, rend_name=1))
                all_atoms = SimulateCoord(all_atoms[0], scale=1)
            output_chemfig_units = reverse_rend_text(all_atoms[0])
            #output_chemfig_units = reverse_chemfig(input_chemfig_str)
            #pdb.set_trace()
            out_text += ["\\chemfig[node style={{scale={}}},atom sep={}em]{{".format(node_scale, atom_seq)] + output_chemfig_units + ["}"]
        else:
            out_text.append(unit)
    return out_text


def main(args):
    inputLines = []
    if os.path.exists(args.input):
        with open(args.input, "r") as fin:
            lines = fin.readlines()
        for line in lines:
            inputLines.append(line.strip())
    elif len(args.input)>1:
        inputLines.append(args.input)
    else:
        s1 = "\chemfig{**6(-=---(-OH)-)}"
        inputLines.append(s1)

    import texlive_rend
    import text_render
    #pdb.set_trace()
    for _id, inputStr in enumerate(inputLines):
        if _id < args.start:
            continue
        if inputStr.startswith("#") or len(inputStr)<2:
            continue
        out_text = reverse_organic_trans(inputStr, debug=True)
        out_text = " ".join(out_text)
        processed_trans = utils.process_trans_for_texlive(out_text)
        rend_img = texlive_rend.texlive_rend(processed_trans)
        cv2.imwrite("./debug/demo.jpg", rend_img)
        # pdb.set_trace()
        re_parse_text = text_render.text_render(processed_trans, debug=True)
        print(inputStr)
        print(processed_trans)
        print(re_parse_text)
        pdb.set_trace()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default="\chemfig{**6(-=---(-OH)-)}")
    parser.add_argument("-start", type=int, default=0)
    args = parser.parse_args()
    main(args)