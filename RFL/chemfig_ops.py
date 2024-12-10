import os, sys
import cv2
from chemfig_struct import *
import math
import numpy
import pdb

def GetAllAtoms(atom:Atom):
    all_atoms = set()
    atom_stack = [atom]
    while len(atom_stack) > 0:
        curAtom = atom_stack.pop()
        if curAtom in all_atoms:
            continue
        if curAtom is None: # add 2022.02.28
            continue
        all_atoms.add(curAtom)
        for bond in curAtom.out_bonds:
            atom_stack.append(bond.end_atom)
        for bond in curAtom.in_bonds:
            atom_stack.append(bond.begin_atom)
    all_atoms = list(all_atoms)
    all_atoms = sorted(all_atoms, key = lambda x:x.name)
    return list(all_atoms)

# some chemfig are too simple and we do not need to use chemfig to express it
def NormFlatChemfig(inAtom:Atom): 
    all_atoms = GetAllAtoms(inAtom)
    out_text = []
    all_atoms = sorted(all_atoms, key=lambda x: x.pos_x)
    last_bond = None
    for atom_id, atom in enumerate(all_atoms):
        all_bonds = atom.in_bonds + atom.out_bonds
        if last_bond is not None:
            if last_bond not in all_bonds:
                return inAtom
            else:
                all_bonds.remove(last_bond)
        if len(all_bonds) <= 0 and atom_id < len(all_atoms) - 1:
            return inAtom
        if len(all_bonds) > 1:
            return inAtom
        if atom_id < len(all_atoms) - 1:
            if len(all_bonds) != 1:
                return inAtom
            new_bond = all_bonds[0]
            if new_bond.begin_atom == atom:
                if new_bond.end_atom != all_atoms[atom_id+1]:
                    return inAtom
                delta = math.fabs(new_bond.m_angle - 0) % 360
                if delta > 180:
                    delta = 360 - delta
                if delta > 15:
                    return inAtom
            else:
                if new_bond.begin_atom != all_atoms[atom_id+1]:
                    return inAtom
                delta = math.fabs(new_bond.m_angle - 180) % 360
                if delta > 180:
                    delta = 360 - delta
                if delta > 15:
                    return inAtom
        else:
            if len(all_bonds) != 0:
                return inAtom
            new_bond = None
        # output text
        if last_bond is not None and last_bond.m_length > 0.25:
            if last_bond.m_type not in bond_in_out_dict:
                return inAtom
            type_str = bond_in_out_dict[last_bond.m_type]
            out_text.append(type_str)
        start_hooks = atom.start_hooks
        end_hooks = atom.end_hooks
        atom.start_hooks = []
        atom.end_hooks = []
        out_text += atom.normed_text()
        atom.start_hooks = start_hooks
        atom.end_hooks = end_hooks
        last_bond = new_bond
    return out_text

_bond_score={"-":0,"=":1,"~":2}
def RemoveDupBonds(rootAtom: Atom):
    atom_stack = [(rootAtom, None)]
    all_atoms = set()
    all_bonds = set()
    bond_dict = {}
    while len(atom_stack):
        curAtom, lastBond = atom_stack.pop()
        if lastBond not in all_bonds:
            all_bonds.add(lastBond)
            if lastBond is not None:
                bond_atom_names = [lastBond.begin_atom.name, lastBond.end_atom.name]
                bond_atom_names = sorted(bond_atom_names, key=lambda x: x)
                bond_key = "-".join(bond_atom_names)
                if bond_key not in bond_dict:
                    bond_dict[bond_key] = []
                bond_dict[bond_key].append(lastBond)
        if curAtom is None:
            continue
        if curAtom in all_atoms:
            continue
        all_atoms.add(curAtom)

        for bond in curAtom.out_bonds:
            if bond.end_atom is not None:
                atom_stack.append((bond.end_atom, bond))
            else:
                raise ValueError("no end atom for bond")
        for bond in curAtom.in_bonds:
            if bond.begin_atom is not None:
                atom_stack.append((bond.begin_atom, bond))
            else:
                raise ValueError("no begin atom for bond")

    for bond_key, bonds in bond_dict.items():
        if len(bonds) <= 1:
            continue
        # pdb.set_trace()
        sorted_bonds = sorted(bonds, key=lambda x:_bond_score.get(x.m_type, 3)*100 + bonds.index(x), reverse=True)
        ref_bond = sorted_bonds[0]
        bond_atom_names = bond_key.split("-")
        for dup_bond in sorted_bonds[1:]:
            assert dup_bond.begin_atom.name in bond_atom_names
            assert dup_bond.end_atom.name in bond_atom_names
            dup_bond.begin_atom.out_bonds.remove(dup_bond)
            dup_bond.end_atom.in_bonds.remove(dup_bond)

def GetCoordRange(all_atoms):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for atom in all_atoms:
        min_x = min(atom.pos_x, min_x)
        min_y = min(atom.pos_y, min_y)
        max_x = max(atom.pos_x, max_x)
        max_y = max(atom.pos_y, max_y)
    return min_x, min_y, max_x, max_y

# connect circle atom to all side atoms
def NormCircleAtom(circleAtom: Atom, base=1e10, all_connect=0):
    if len(circleAtom.out_bonds) +  len(circleAtom.in_bonds) != 1:
        return

    if len(circleAtom.out_bonds) == 1: #circle -> begin
        old_branch_info = circleAtom.out_bonds[0].branch_info
        begin_atom = circleAtom.out_bonds[0].end_atom
        origin_length = circleAtom.out_bonds[0].m_length
        begin_atom.in_bonds.remove(circleAtom.out_bonds[0])
        circleAtom.out_bonds.clear()
    else: #circle <- begin
        old_branch_info = circleAtom.in_bonds[0].branch_info
        begin_atom = circleAtom.in_bonds[0].begin_atom
        origin_length = circleAtom.in_bonds[0].m_length
        begin_atom.out_bonds.remove(circleAtom.in_bonds[0])
        circleAtom.in_bonds.clear()

    ref_angle = math.atan2(-begin_atom.pos_y + circleAtom.pos_y, begin_atom.pos_x - circleAtom.pos_x) * 180.0 / math.pi
    ref_length = math.sqrt(math.pow(begin_atom.pos_y - circleAtom.pos_y, 2) + math.pow(begin_atom.pos_x - circleAtom.pos_x, 2))
    visited = set()
    path = []
    
    def dfs_find(curAtom, lastBond=None, visited=[]):
        if curAtom in visited:
            return [visited]
        # print("step into:", curAtom)
        cur_visited = visited + [curAtom]
        cur_angle = math.atan2(-curAtom.pos_y + circleAtom.pos_y, curAtom.pos_x - circleAtom.pos_x) * 180.0 / math.pi
        ret = []
        cand_bonds = []
        for bond in curAtom.out_bonds + curAtom.in_bonds:
            next_atom = bond.end_atom if bond.end_atom != curAtom else bond.begin_atom
            if len(visited)>0 and next_atom == visited[-1]:
                continue
            if next_atom.m_text == "\\circle" or isinstance(next_atom, CircleAtom):
                continue
            next_angle = math.atan2(-next_atom.pos_y + circleAtom.pos_y, next_atom.pos_x - circleAtom.pos_x) * 180.0 / math.pi
            next_length = math.sqrt(math.pow(next_atom.pos_y - circleAtom.pos_y, 2) + math.pow(next_atom.pos_x - circleAtom.pos_x, 2))
            delta_length = math.fabs(next_length - ref_length)
            cand_bonds.append((bond, next_atom, next_angle, next_length, delta_length))
        cand_bonds = sorted(cand_bonds, key=lambda x:x[-1])
        for bond, next_atom, next_angle, next_length, delta_length in cand_bonds:
            if (next_angle-cur_angle) % 360 >= 180:
                # print("skip1:", next_atom)
                continue
            if math.fabs(next_length - ref_length) > ref_length * 1.0: #changed to 0.5 at 2022.02.22 by haowu16
                # print("skip2:", next_atom)
                continue
            # print("before dfs:", next_atom)
            ret += dfs_find(next_atom, bond, cur_visited)
        return ret
    
    # from image_render import rend
    # rend(begin_atom, scale=100, rend_name=1)
    det_results = dfs_find(begin_atom, None, [])
    if len(det_results) <= 0:
        det_results = [[begin_atom]]

    ring_atoms = det_results[0]
    ring_atoms = sorted(ring_atoms, key = lambda x:x.pos_x*base - x.pos_y)
    selected_atom = ring_atoms[0]

    for selected_atom in ring_atoms:
        new_bond = Bond("-:")
        new_bond.branch_info = old_branch_info
        new_bond.begin_atom = selected_atom
        new_bond.end_atom = circleAtom
        new_bond.m_angle = math.atan2( - new_bond.end_atom.pos_y + new_bond.begin_atom.pos_y, new_bond.end_atom.pos_x - new_bond.begin_atom.pos_x) * 180.0 / math.pi
        new_bond.m_length = origin_length
        selected_atom.out_bonds.append(new_bond)
        circleAtom.in_bonds.append(new_bond)
        if not all_connect:
            break

    return

def NormAllCircleAtom(rootAtom:Atom, all_connect=0):
    all_atoms = GetAllAtoms(rootAtom)
    min_x, min_y, max_x, max_y = GetCoordRange(all_atoms)
    max_value = max(max_y-min_x, max_y-min_y)
    for atom in all_atoms:
        if isinstance(atom, CircleAtom) or atom.m_text == "\\circle":
            NormCircleAtom(atom, base=max_value, all_connect=all_connect)
    return all_atoms

# remove hooks in atoms and add bonds
def ConnectDistantAtoms(rootAtom:Atom, scale=1.0, bond_dict=[]):
    all_atoms = GetAllAtoms(rootAtom)
    hook_dict = {} #{"<hookname>:{"start":(<Atom>,<hook>), "end":[(<Atom1>,hook1), (<Atom2>,hook2)]}"}
    for atom in all_atoms:
        for start_hook in atom.start_hooks:
            hook_name = start_hook.m_hookname
            if hook_name not in hook_dict:
                hook_dict[hook_name] = {"start":None, "end":[]}
            if hook_dict[hook_name]["start"] is not None:
                # pdb.set_trace()
                raise ValueError("multiple start hook detect!")
            hook_dict[hook_name]["start"] = (atom, start_hook)
        for end_hook in atom.end_hooks:
            hook_name = end_hook.m_hookname
            if hook_name not in hook_dict:
                hook_dict[hook_name] = {"start":None, "end":[]}
            hook_dict[hook_name]["end"].append((atom, end_hook))
    
    #create bond
    for hook_name in hook_dict:
        start_atom, start_hook = hook_dict[hook_name]["start"]
        pre_start_atom = start_atom
        for end_atom, hook in hook_dict[hook_name]["end"]:
            start_atom = pre_start_atom
            angle = math.atan2(-end_atom.pos_y + start_atom.pos_y, end_atom.pos_x - start_atom.pos_x) * 180.0 / math.pi 
            length = math.sqrt(math.pow(end_atom.pos_x - start_atom.pos_x, 2) + math.pow(end_atom.pos_y - start_atom.pos_y, 2) ) / scale
            new_bond = Bond(b_type=hook.m_bondtype)
            new_bond.m_angle = angle
            new_bond.m_length = length
            start_atom, end_atom = end_atom, start_atom # 修改回连的连接方向,确保顺序相同
            new_bond.begin_atom = start_atom
            new_bond.end_atom = end_atom
            end_atom.in_bonds.append(new_bond)
            start_atom.out_bonds.append(new_bond)
            # end_atom.end_hooks.remove(hook)
            start_atom.end_hooks.remove(hook) # 因为交换了start, end.所以这里也得交换
            # 遍历bond_dict将Distancehook替换为new_bond
            find = False
            for i, unit in enumerate(bond_dict):
                if unit[0] == hook:
                    bond_dict[i][0] = new_bond
                    find = True
                    break
            if not find:
                raise ValueError("can't find the match hook and bond")
            # bond_dict[new_bond] = bond_dict.pop(hook) # 将hook:len -> bond:len
        # start_atom.start_hooks.remove(start_hook)
            
        end_atom.start_hooks.remove(start_hook) # 因为交换了start, end.所以这里也得交换
        
    return
    
def RemoveDupAtoms(rootAtom:Atom, th=0.01):
    def JudgeCorrd(curAtom, all_atoms, th=0.01):
        for atom in all_atoms:
            distance = math.sqrt(math.pow(atom.pos_x - curAtom.pos_x, 2) + math.pow(atom.pos_y - curAtom.pos_y, 2))
            if distance < th:
                return atom
        return None
    atom_stack = [(rootAtom, None)]
    all_atoms = []
    while len(atom_stack):
        curAtom, lastBond = atom_stack.pop()
        if curAtom is None:
            continue
        if curAtom in all_atoms:
            continue
        duplicateAtom = JudgeCorrd(curAtom, all_atoms, th=th)
        if duplicateAtom is not None:
            if curAtom.m_text != "" and duplicateAtom.m_text!="":
                all_atoms.append(curAtom)
            else:
                for bond in curAtom.in_bonds:
                    if bond.begin_atom == duplicateAtom:
                        continue
                    bond.end_atom = duplicateAtom
                    duplicateAtom.in_bonds.append(bond)
                for bond in curAtom.out_bonds:
                    if bond.end_atom == duplicateAtom:
                        continue
                    bond.begin_atom = duplicateAtom
                    duplicateAtom.out_bonds.append(bond)

                new_in_bonds = []
                for bond in duplicateAtom.in_bonds:
                    if bond.begin_atom == curAtom:
                        continue
                    new_in_bonds.append(bond)
                duplicateAtom.in_bonds = new_in_bonds
                
                new_out_bonds = []
                for bond in duplicateAtom.out_bonds:
                    if bond.end_atom == curAtom:
                        continue
                    new_out_bonds.append(bond)
                duplicateAtom.out_bonds = new_out_bonds


                curAtom.in_bonds.clear()
                curAtom.out_bonds.clear()
                # add 2022.01.14, copy hooks also
                for start_hook in curAtom.start_hooks:
                    duplicateAtom.start_hooks.append(start_hook)
                for end_hook in curAtom.end_hooks:
                    duplicateAtom.end_hooks.append(end_hook)
                curAtom.start_hooks.clear() # add 2022.02.28
                curAtom.end_hooks.clear() #
                if duplicateAtom.m_text == "" and curAtom.m_text != "":#fixed bug where atom text may miss
                    duplicateAtom.m_text = curAtom.m_text
                curAtom = duplicateAtom
        else:
            all_atoms.append(curAtom)
        # continue traverse
        for bond in curAtom.out_bonds:
            if bond.end_atom is not None:
                atom_stack.append((bond.end_atom, bond))
            else:
                end_atom = Atom("")
                bond.end_atom = end_atom
                end_atom.in_bonds.append(bond)
                atom_stack.append((bond.end_atom, bond))
        for bond in curAtom.in_bonds:
            if bond.begin_atom is not None:
                atom_stack.append((bond.begin_atom, bond))
            else:
                begin_atom = Atom("")
                bond.begin_atom = begin_atom
                begin_atom.out_bonds.append(bond)
                atom_stack.append((bond.begin_atom, bond))
    return all_atoms

# gen coord for atoms and do some change
def SimulateCoord(rootAtom: Atom, scale=1):
    atom_stack = [(rootAtom, None)]
    all_atoms = []
    while len(atom_stack):
        curAtom, lastBond = atom_stack.pop()
        if curAtom is None:
            continue
        if curAtom in all_atoms:
            continue
        if lastBond is None:
            curAtom.pos_x = 0
            curAtom.pos_y = 0
        else:
            if lastBond.end_atom == curAtom:
                lastAtom = lastBond.begin_atom
                curAtom.pos_x = lastAtom.pos_x + lastBond.m_length * scale * math.cos(lastBond.m_angle * math.pi / 180)
                curAtom.pos_y = lastAtom.pos_y - lastBond.m_length * scale * math.sin(lastBond.m_angle * math.pi / 180)
            elif lastBond.begin_atom == curAtom:
                lastAtom = lastBond.end_atom
                curAtom.pos_x = lastAtom.pos_x - lastBond.m_length * scale * math.cos(lastBond.m_angle * math.pi / 180)
                curAtom.pos_y = lastAtom.pos_y + lastBond.m_length * scale * math.sin(lastBond.m_angle * math.pi / 180)

        all_atoms.append(curAtom)

        for bond in curAtom.out_bonds:
            if bond.end_atom is not None:
                atom_stack.append((bond.end_atom, bond))
            else:
                end_atom = Atom("")
                bond.end_atom = end_atom
                end_atom.in_bonds.append(bond)
                atom_stack.append((bond.end_atom, bond))
        for bond in curAtom.in_bonds:
            if bond.begin_atom is not None:
                atom_stack.append((bond.begin_atom, bond))
            else:
                begin_atom = Atom("")
                bond.begin_atom = begin_atom
                begin_atom.out_bonds.append(bond)
                atom_stack.append((bond.begin_atom, bond))
    return all_atoms
