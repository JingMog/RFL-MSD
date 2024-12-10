import os, sys
import cv2
from chemfig_struct import *
from chemfig_ops import *
from chemfig_parser import *
import image_render
import utils
import math
import numpy
import pdb
from Tool_Formula.latex_norm.transcription import parse_transcription
import Tool_Formula.latex_norm.transcription as trans
import argparse

def dfs_visited(cur_atom, last_bond=None, angle=0, visited=[], out_text=[], info_arr=[], withAngle=1, angle_step=15, branch_de_amb=0, branch_arr=[], bond_dict=[]):
    if last_bond in visited:
        return

    if last_bond is not None:
        if not cur_atom in visited:
            if last_bond.m_type is not None and last_bond.m_type != "" and (last_bond.m_type != "-:" or isinstance(cur_atom, CircleAtom) or cur_atom.m_text == "\\circle"):
                out_text.append(last_bond.m_type)
                if withAngle > 0:
                    if angle_step is not None:
                        normed_angle = int((angle+angle_step/2.0) / angle_step) * angle_step
                        normed_angle = normed_angle % 360
                    else:
                        normed_angle = float("{:.2f}".format(angle))
                        normed_angle = normed_angle % 360
                    out_text[-1] += "[:{}]".format(normed_angle)
                    bond_dict.append([last_bond, len(out_text)-1])
                    # bond_dict[last_bond] = len(out_text)-1 # 维护bond_dictc
                branch_arr.append(last_bond.branch_info) # 获取分支键与环相连的相对位置
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
            
            bond_dict.append([last_bond, ])
            

        visited.append(last_bond)

    if cur_atom in visited:
        return

    if cur_atom is not None:
        out_text.append(cur_atom)
        visited.append(cur_atom)
        branch_arr.append(None)
        # ring_branch_arr.append(None)

    all_bonds = []
    for bond, angle in cur_atom.all_bonds:
        if cur_atom == bond.end_atom:
            all_bonds.append((bond, angle, bond.begin_atom))
        elif cur_atom == bond.begin_atom:
            all_bonds.append((bond, angle, bond.end_atom))
        else:
            raise ValueError("atom bond not connect")

    pairs = []
    last_ind = -1

    for child_id, (bond, angle, next_atom) in enumerate(all_bonds):
        if branch_de_amb > 0:
            out_text.append("branch(")
            branch_arr.append(None)
            # ring_branch_arr.append(None)
        else:
            out_text.append("(")
            branch_arr.append(None)
            # ring_branch_arr.append(None)
        begin_ind = len(out_text) - 1
        dfs_visited(next_atom, bond, angle, visited, out_text, info_arr, withAngle, angle_step, branch_de_amb=branch_de_amb, branch_arr=branch_arr, bond_dict=bond_dict)
        if branch_de_amb > 0:
            out_text.append("branch)")
            branch_arr.append(None)
            # ring_branch_arr.append(None)
        else:
            out_text.append(")")
            branch_arr.append(None)
            # ring_branch_arr.append(None)
        end_ind = len(out_text) - 1
        pairs.append((begin_ind, end_ind))
        if end_ind == begin_ind + 1:
            info_arr += [begin_ind, end_ind]
            last_ind = len(pairs) - 2
        else:
            last_ind = len(pairs) - 1

    if len(pairs) > 0 and last_ind >= 0:
        info_arr += list(pairs[last_ind])



def pre_process(rootAtom, scale = 1, preprocess=True, bond_dict=[]):
    all_atoms = SimulateCoord(rootAtom, scale=scale)
    if preprocess:
        all_atoms = RemoveDupAtoms(all_atoms[0], th=scale*0.01)
        RemoveDupBonds(all_atoms[0])
    ConnectDistantAtoms(all_atoms[0], bond_dict=bond_dict) # 这里会处理DistanceHook,创建回连的Bond
    all_atoms = NormAllCircleAtom(all_atoms[0])
    return all_atoms

def rend_text(rootAtom, withAngle=1, branch_de_amb=0):
    # all_atoms = SimulateCoord(rootAtom, scale=scale)
    # all_atoms = RemoveDupAtoms(all_atoms[0], th=scale*0.01)
    # RemoveDupBonds(all_atoms[0])
    # all_atoms = NormAllCircleAtom(all_atoms[0])
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
    dfs_visited(start_atom, None, 0, visited, out_seq, info_arr, withAngle, branch_de_amb=branch_de_amb)

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
            # if element.name=="Atom_0":
            #     pdb.set_trace()
            if isinstance(element, CircleAtom):
                out_text += ["\\circle"]
            else:
                out_text += element.normed_text()
            element.start_hooks = [] # must clear hooks here, other wise it will affect following operation!!!
            element.end_hooks = [] #
        elif isinstance(element, Bond):
            last_bond = element
            cur_atom = last_bond.end_atom
            angle = math.atan2(last_bond.end_atom.pos_y - last_bond.begin_atom.pos_y, last_bond.end_atom.pos_x - last_bond.begin_atom.pos_x)
            if last_bond.m_type is not None and last_bond.m_type != "" and (last_bond.m_type != "-:" or isinstance(cur_atom, CircleAtom) or cur_atom.m_text == "\\circle"):
                out_text.append(last_bond.m_type)
            if withAngle > 0:
                normed_angle = int((angle+7.5) / 15.0) * 15
                normed_angle = normed_angle % 360
                out_text.append("[:{}]".format(normed_angle))



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
    # if atom_count != len(all_atoms):
    #     print("------------------{} vs {} ---------------------".format(len(all_atoms), atom_count))
    #     print(sorted(all_atoms, key=lambda x:x.name))
    #     visited_atoms = []
    #     for element in visited:
    #         if isinstance(element, Atom):
    #             visited_atoms.append(element)
    #     print(sorted(visited_atoms, key = lambda x:x.name)) 
    #     print("-------------------------------------------------------")
    return out_text

# interface for text render
def text_render(in_text, debug=False, branch_de_amb=0):
    _, rep_dict, text = utils.replace_chemfig(in_text)
    new_text = parse_transcription(text, simple_trans=True)

    new_rep_dict = {}
    for key, value in rep_dict.items():
        chemfig_str = value.strip()[8:].strip()[1:-1]
        rootAtom, _ = chemfig_parse(chemfig_str, echo=debug)
        all_atoms = pre_process(rootAtom, scale=1)
        chemfig_units = rend_text(all_atoms[0], branch_de_amb=branch_de_amb)
        chemfig_units = ["\chemfig", "{"] + chemfig_units + ["}"]
        new_rep_dict[key] = chemfig_units
        if debug:
            
            all_atoms = SimulateCoord(all_atoms[0], scale=100)
            img_key = key.replace("\\chem", "")
            cv2.imwrite("./debug/rend_{}.jpg".format(img_key), image_render.rend_atoms(all_atoms, scale=100, rend_name=0))
            cv2.imwrite("./debug/rend_name_{}.jpg".format(img_key), image_render.rend_atoms(all_atoms, scale=100, rend_name=1))

    out_text = []
    for unit in new_text:
        if unit in new_rep_dict:
            out_text += new_rep_dict[unit]
        else:
            out_text.append(unit)
    out_text = " ".join(out_text)
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
    for _id, inputStr in enumerate(inputLines):
        if _id < args.start:
            continue
        if inputStr.startswith("#") or len(inputStr)<2:
            continue
        out_text = text_render(inputStr, debug=True)
        rend_img = texlive_rend.texlive_rend("$"+inputStr+"$")
        cv2.imwrite("./debug/demo.jpg", rend_img)
        print(inputStr)
        print(out_text)
        pdb.set_trace()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", type=str, default="\chemfig{**6(-=---(-OH)-)}")
    parser.add_argument("-start", type=int, default=0)
    args = parser.parse_args()
    main(args)

