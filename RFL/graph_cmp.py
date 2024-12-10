import os, sys
import pdb
import argparse
import Levenshtein
import chemfig_struct
import chemfig_ops
import chemfig_parser
import text_render
from chemfig_struct import *
import utils
import tqdm
import cv2
import math
import numpy

# interface
def compare_graph(rootA: Atom, rootB: Atom):
    # pdb.set_trace()
    res_text, textA, textB = match_text(rootA, rootB)
    res_graph = match_graph(rootA, rootB)
    if res_text == 1 and res_graph == 1 or True:
        # print("res_text={}".format(res_text))
        # print("res_graph={}".format(res_graph))
        # import image_render
        # cv2.imwrite("a.jpg", image_render.rend(rootA, scale=100))
        # cv2.imwrite("a2.jpg", image_render.rend(rootA, scale=100, rend_name=1))

        # cv2.imwrite("b.jpg", image_render.rend(rootB, scale=100))
        # cv2.imwrite("b2.jpg", image_render.rend(rootB, scale=100, rend_name=1))
        # print("textA: {}".format(textA))
        # print("textB: {}".format(textB))
        # pdb.set_trace()
        pass
    if res_text == 0 or res_graph == 0:
        return 0
    else:
        return 1

def match_text(rootA: Atom, rootB: Atom):
    arr_A = text_render.rend_text(rootA, 1)
    arr_B = text_render.rend_text(rootB, 1)

    textA = " ".join(arr_A)
    textB = " ".join(arr_B)
    textA_noAngle = " ".join(utils.removeAngle(arr_A))
    textB_noAngle = " ".join(utils.removeAngle(arr_B))
    if textA_noAngle == textB_noAngle:
        # cv2.imwrite("a.jpg", chemfig_render.rend(rootA, scale=100))
        # cv2.imwrite("a2.jpg", chemfig_render.rend(rootA, scale=100, rend_name=1))
        # cv2.imwrite("b.jpg", chemfig_render.rend(rootB, scale=100))
        # print("txtA: {}".format(txtA))
        # print("txtB: {}".format(txtB))
        # print("textA: {}".format(textA))
        # print("textB: {}".format(textB))
        # # print(os.popen("cat show.txt").readlines()[0])
        # pdb.set_trace()
        return 0, textA, textB
    else:
        # with open("debug.txt", "a") as fout:
        #     fout.write("txtA: {}\n".format(txtA))
        #     fout.write("txtA: {}\n".format(txtB))
        #     fout.write("textA: {}\n".format(textA))
        #     fout.write("textB: {}\n".format(textB))
        #     fout.write("\n")
        # cv2.imwrite("a.jpg", chemfig_render.rend(rootA, scale=100))
        # cv2.imwrite("a2.jpg", chemfig_render.rend(rootA, scale=100, rend_name=1))
        # cv2.imwrite("b.jpg", chemfig_render.rend(rootB, scale=100))
        # cv2.imwrite("b2.jpg", chemfig_render.rend(rootB, scale=100, rend_name=1))
        # print("txtA: {}".format(txtA))
        # print("txtB: {}".format(txtB))
        # print("textA: {}".format(textA))
        # print("textB: {}".format(textB))
        # print(os.popen("cat show.txt").readlines()[0])
        # pdb.set_trace()

        # print("textA: {}".format(textA))
        # print("textB: {}".format(textB))
        # print("-----------------------")
        return 1, textA, textB

def cmp_bond_type(bondA:Bond, bondB:Bond):
    bond_same = False
    if bondA.m_type in chemfig_struct.directed_bond_types and bondB.m_type in chemfig_struct.directed_bond_types:
        bond_delta_angle = math.fabs(bondA.m_angle - bondB.m_angle)
        if bond_delta_angle > 180:
            bond_delta_angle = 360 - bond_delta_angle
        # if bond_delta_angle > 90:
        #     bond_same = (bondA.m_type == chemfig_struct.directed_bond_types[bondB.m_type])
        # else:
        #     bond_same = (bondA.m_type == bondB.m_type)
        bond_same = ((bondA.m_type == chemfig_struct.directed_bond_types[bondB.m_type]) or (bondA.m_type == bondB.m_type))
    else:
        bond_same = (bondA.m_type == bondB.m_type)
    return bond_same

def compare_atom_dist(atomA: Atom, atomB: Atom):
    #compare text
    ed_ops = utils.cal_edit_ops(atomA.normed_text(), atomB.normed_text())
    ed_dist = len(ed_ops)
    #compare degree
    degree_dist = math.fabs(atomA.degree - atomB.degree)
    #compare content
    if degree_dist > 0:
        content_dist = 1e10
    contentA = atomA.content_arr
    contentB = atomB.content_arr
    if atomA.degree < atomB.degree:
        shift_content = contentB
        cmp_content = contentA
    else:
        shift_content = contentA
        cmp_content = contentB
    min_shift_content = None
    min_content_dist = math.inf
    for shift in range(0, len(shift_content)):
        ref_content = shift_content[shift:] + shift_content[:shift]
        cur_content_dist = 0
        for ind in range(len(cmp_content)):
            ref_item = ref_content[ind]  #angle, bond_type, tgt_text, tgt_degree
            cmp_item = cmp_content[ind]
            # cur_content_dist += math.fabs(ref_item[0] - cmp_item[0])  #angle
            delta_angle = (ref_item[0] - cmp_item[0])%360
            if delta_angle > 180:
                delta_angle = 360 - delta_angle
            cur_content_dist += delta_angle  #angle
            cur_content_dist += (not cmp_bond_type(ref_item[1], cmp_item[1])) * 10.0 # update @2022.07.20 haowu16
            cur_content_dist += math.fabs(ref_item[2].degree - cmp_item[2].degree) * 30  # degree
        if cur_content_dist < min_content_dist:
            min_content_dist = cur_content_dist
            min_shift_content = ref_content
    content_dist = min_content_dist

    total_distance = ed_dist*10 + degree_dist*30 + content_dist
    if atomA.degree < atomB.degree:
        match_result = (contentA, min_shift_content)
    else:
        match_result = (min_shift_content, contentB)
    return total_distance, match_result

def GuidedWalk(start_atomA, start_atomB, debug=False):
    atom_stackA = [(None, start_atomA)]
    atom_stackB = [(None, start_atomB)]
    visitedA = set()
    visitedB = set()
    while len(atom_stackA) > 0:
        cur_BondA, cur_atomA = atom_stackA.pop()
        cur_BondB, cur_atomB = atom_stackB.pop()
        if cur_atomA in visitedA:
            if cur_atomB not in visitedB:
                if debug:
                    pdb.set_trace()
                return False
            else:
                continue
        if cur_atomB in visitedB:
            return False
        if debug:
            print("A={} B={}".format(cur_atomA.name, cur_atomB.name))
        atomA_text = cur_atomA.normed_text() if not isinstance(cur_atomA, CircleAtom) else ["\\circle"]
        atomB_text = cur_atomB.normed_text() if not isinstance(cur_atomB, CircleAtom) else ["\\circle"]
        atomA_text = utils.post_process(atomA_text)
        atomB_text = utils.post_process(atomB_text)
                
        if atomA_text != atomB_text:
            if debug:
                pdb.set_trace()
            return False
        visitedA.add(cur_atomA)
        visitedB.add(cur_atomB)
        contentA = cur_atomA.content_arr
        contentB = cur_atomB.content_arr
        if len(contentA) != len(contentB):
            if debug:
                pdb.set_trace()
            return False
        
        min_shift_contentB = []
        min_content_dist = math.inf
        for shift in range(0, len(contentB)):
            ref_content = contentB[shift:] + contentB[:shift]
            cur_content_dist = 0
            for ind in range(len(contentA)):
                ref_item = ref_content[ind]  #angle, bond_type, tgt_text, tgt_degree
                cmp_item = contentA[ind]
                # cur_content_dist += math.fabs(ref_item[0] - cmp_item[0])  #angle
                delta_angle = (ref_item[0] - cmp_item[0])%360
                if delta_angle > 180:
                    delta_angle = 360 - delta_angle
                cur_content_dist += delta_angle  #angle
                cur_content_dist += (not cmp_bond_type(ref_item[1], cmp_item[1])) * 10.0  #bond_type
                cur_content_dist += math.fabs(ref_item[2].degree - cmp_item[2].degree) * 30  # degree
            if cur_content_dist < min_content_dist:
                min_content_dist = cur_content_dist
                min_shift_contentB = ref_content
        for itemA, itemB in zip(contentA, min_shift_contentB):
            #if itemA[1].m_type != itemB[1].m_type:
            if not cmp_bond_type(itemA[1], itemB[1]):
                if debug:
                    pdb.set_trace()
                return False
            atom_stackA.append((itemA[1], itemA[2]))
            atom_stackB.append((itemB[1], itemB[2]))
    return True

def pre_process_for_cmp(rootAtom:Atom):
    all_atoms = chemfig_ops.SimulateCoord(rootAtom, scale=1)
    all_atoms = chemfig_ops.RemoveDupAtoms(all_atoms[0])
    chemfig_ops.RemoveDupBonds(all_atoms[0])
    chemfig_ops.ConnectDistantAtoms(all_atoms[0])
    all_atoms = chemfig_ops.NormAllCircleAtom(all_atoms[0])
    return all_atoms

def match_graph(rootA: Atom, rootB: Atom):
    a_atoms = chemfig_ops.NormAllCircleAtom(rootA, all_connect=1)
    b_atoms = chemfig_ops.NormAllCircleAtom(rootB, all_connect=1)
    
    a_name2idx = dict([(atom.name, ind) for ind, atom in enumerate(a_atoms)])
    b_name2idx = dict([(atom.name, ind) for ind, atom in enumerate(b_atoms)])

    #calc atom attribute [degree, content] content_arr
    for child_atom in a_atoms + b_atoms:
        child_atom.degree = len(child_atom.in_bonds + child_atom.out_bonds)
        child_atom.content_arr = []  #angle, bond_type, tgt_text, tgt_degree
        for in_bond in child_atom.in_bonds + child_atom.out_bonds:
            tgt_atom = in_bond.begin_atom if in_bond.begin_atom != child_atom else in_bond.end_atom
            angle = math.atan2(-tgt_atom.pos_y + child_atom.pos_y, tgt_atom.pos_x - child_atom.pos_x) * 180.0 / math.pi
            angle = angle % 360
            child_atom.content_arr.append((angle, in_bond, tgt_atom))
        child_atom.content_arr = sorted(child_atom.content_arr, key=lambda x: x[0])

    #compare atom
    dist_mat = numpy.zeros((len(a_atoms), len(b_atoms)), dtype=numpy.float)  #lenA, lenB
    atom_match_results = []
    for child_idA, child_atomA in enumerate(a_atoms):
        cur_dist_arr = []
        perfect_num = 0
        for child_idB, child_atomB in enumerate(b_atoms):
            total_distance, match_result = compare_atom_dist(child_atomA, child_atomB)
            cur_dist_arr.append((child_idB, total_distance, match_result))
            dist_mat[child_idA, child_idB] = total_distance
            if total_distance < 1e-6:
                perfect_num += 1
        cur_dist_arr = sorted(cur_dist_arr, key=lambda x: x[1])
        atom_match_results.append((child_idA, cur_dist_arr, perfect_num))
    
    # select pairs
    a_perfect_match_num = (dist_mat==0).sum(1, keepdims=True)
    b_perfect_match_num = (dist_mat==0).sum(0, keepdims=True)
    # a_match_dist = (a_perfect_match_num == 1)*0 + (a_perfect_match_num > 1)*a_perfect_match_num + (a_perfect_match_num == 0)*1000
    # b_match_dist = (b_perfect_match_num == 1)*0 + (b_perfect_match_num > 1)*b_perfect_match_num + (b_perfect_match_num == 0)*1000
    match_dist = ~((a_perfect_match_num == 1)&(b_perfect_match_num == 1))
    match_dist = match_dist.astype("float")

    cond_match_dist = match_dist * (dist_mat == 0) * 1000 + (dist_mat > 0) * 2000
    fused_dist = dist_mat + cond_match_dist

    pairs = [(i,j) for j in range(len(b_atoms)) for i in range(len(a_atoms))]
    pairs = sorted(pairs, key = lambda x:fused_dist[x[0],x[1]])
    # for a_ind, b_ind in pairs[0:10]:
    #     print("{}-{}: {} + {}".format(a_atoms[a_ind].name, b_atoms[b_ind].name, dist_mat[a_ind, b_ind], cond_match_dist[a_ind, b_ind]))
    #pdb.set_trace()
    
    max_try_count = 3
    cur_try_count = 0
    for indA, indB in pairs:
        start_atomA = a_atoms[indA] #reference
        start_atomB = b_atoms[indB]
        cmp_result = GuidedWalk(start_atomA, start_atomB, False)
        cur_try_count += 1
        if cmp_result is False:
            if cur_try_count > max_try_count:
                return 1
            else:
                continue
        else:
            return 0
    return 1

