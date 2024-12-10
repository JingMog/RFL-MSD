import os, sys
import cv2
from chemfig_struct import *
from chemfig_ops import *
from chemfig_parser import *
import utils
import math
import numpy
import pdb
from Tool_Formula.latex_norm.transcription import parse_transcription
import Tool_Formula.latex_norm.transcription as trans
import argparse

def pre_process_for_image_render(rootAtom, scale = 1):
    all_atoms = SimulateCoord(rootAtom, scale=scale)
    all_atoms = RemoveDupAtoms(all_atoms[0], th=scale*0.01)
    RemoveDupBonds(all_atoms[0])
    ConnectDistantAtoms(all_atoms[0])
    all_atoms = NormAllCircleAtom(all_atoms[0])
    return all_atoms

def rend_atoms(all_atoms, scale=100, rend_name=0):
    min_x, min_y, max_x, max_y = GetCoordRange(all_atoms)
    margin_h = int(scale)
    margin_w = int(scale)
    img_height = max_y - min_y + 2*margin_h
    img_width = max_x - min_x + 2*margin_w
    img_height = int(img_height)
    img_width = int(img_width)
    out_img = numpy.ones((img_height, img_width, 3), dtype="uint8") * 255

    for atom in all_atoms:
        atom.pos_x = atom.pos_x - min_x + margin_w
        atom.pos_y = atom.pos_y - min_y + margin_h
        #print(atom)

    rootAtom = all_atoms[0] 
    
    # draw bonds first
    atom_stack = [(rootAtom, None)]
    visited_atoms = []
    while len(atom_stack):
        curAtom, lastBond = atom_stack.pop()
        if curAtom is None:
            continue
        extend = True
        if curAtom in visited_atoms:
            extend = False
        else:
            visited_atoms.append(curAtom)
        tgt_x = curAtom.pos_x
        tgt_y = curAtom.pos_y

        if lastBond is not None:
            begin_pos = (lastBond.begin_atom.pos_x, lastBond.begin_atom.pos_y)
            end_pos = (lastBond.end_atom.pos_x, lastBond.end_atom.pos_y)
            shift = 3
            if lastBond.m_type == "=":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                cv2.line(out_img, (int(begin_pos[0] + offset_x), int(begin_pos[1] + offset_y)), (int(end_pos[0] + offset_x), int(end_pos[1] + offset_y)), (255, 0, 255), 1)
                cv2.line(out_img, (int(begin_pos[0] - offset_x), int(begin_pos[1] - offset_y)), (int(end_pos[0] - offset_x), int(end_pos[1] - offset_y)), (255, 0, 255), 1)
            elif lastBond.m_type == "~":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                cv2.line(out_img, (int(begin_pos[0] + offset_x), int(begin_pos[1] + offset_y)), (int(end_pos[0] + offset_x), int(end_pos[1] + offset_y)), (255, 0, 0), 1)
                cv2.line(out_img, (int(begin_pos[0] - offset_x), int(begin_pos[1] - offset_y)), (int(end_pos[0] - offset_x), int(end_pos[1] - offset_y)), (255, 0, 0), 1)
                cv2.line(out_img, (int(begin_pos[0]), int(begin_pos[1])), (int(end_pos[0]), int(end_pos[1])), (255, 0, 0), 1)
            elif lastBond.m_type == "-":
                cv2.line(out_img, (int(begin_pos[0]), int(begin_pos[1])), (int(end_pos[0]), int(end_pos[1])), (255, 0, 0), 1)
            elif lastBond.m_type == "-:":
                cv2.line(out_img, (int(begin_pos[0]), int(begin_pos[1])), (int(end_pos[0]), int(end_pos[1])), (200, 200, 200), 1, cv2.LINE_AA)
            elif lastBond.m_type == "<":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                contours = [(begin_pos[0], begin_pos[1])]
                contours.append((end_pos[0] + offset_x, end_pos[1] + offset_y))
                contours.append((end_pos[0] - offset_x, end_pos[1] - offset_y))
                contours = np.array([contours,]).astype("int32")
                cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=-1)
            elif lastBond.m_type == ">":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                contours = [(end_pos[0], end_pos[1])]
                contours.append((begin_pos[0] + offset_x, begin_pos[1] + offset_y))
                contours.append((begin_pos[0] - offset_x, begin_pos[1] - offset_y))
                contours = np.array([contours,]).astype("int32")
                cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=-1)
            elif lastBond.m_type == ">|":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                contours = [(end_pos[0], end_pos[1])]
                contours.append((begin_pos[0] + offset_x, begin_pos[1] + offset_y))
                contours.append((begin_pos[0] - offset_x, begin_pos[1] - offset_y))
                contours = np.array([contours,]).astype("int32")
                cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=1)
            elif lastBond.m_type == "<|":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                contours = [(begin_pos[0], begin_pos[1])]
                contours.append((end_pos[0] + offset_x, end_pos[1] + offset_y))
                contours.append((end_pos[0] - offset_x, end_pos[1] - offset_y))
                contours = np.array([contours,]).astype("int32")
                cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=1)
            elif lastBond.m_type == ">:":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                shift = 6
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                contours = [(end_pos[0], end_pos[1])]
                contours.append((begin_pos[0] + offset_x, begin_pos[1] + offset_y))
                contours.append((begin_pos[0] - offset_x, begin_pos[1] - offset_y))
                _, begin_pos_up, begin_pos_down = contours
                sub_line_num = 5
                step = 1.0/(sub_line_num-1)
                for sub_line_id in range(sub_line_num):
                    ratio = sub_line_id * step
                    pos1 = (begin_pos_up[0]*ratio+end_pos[0]*(1.0-ratio), begin_pos_up[1]*ratio+end_pos[1]*(1.0-ratio))
                    pos2 = (begin_pos_down[0]*ratio+end_pos[0]*(1.0-ratio), begin_pos_down[1]*ratio+end_pos[1]*(1.0-ratio))
                    cv2.line(out_img, (int(pos1[0]), int(pos1[1])), (int(pos2[0]), int(pos2[1])), (0, 0, 255), 3)
                # contours = np.array([contours,]).astype("int32")
                # cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=1)
            elif lastBond.m_type == "<:":
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                shift = 6
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                contours = [(begin_pos[0], begin_pos[1])]
                contours.append((end_pos[0] + offset_x, end_pos[1] + offset_y))
                contours.append((end_pos[0] - offset_x, end_pos[1] - offset_y))
                _, end_pos_up, end_pos_down = contours
                sub_line_num = 5
                step = 1.0/(sub_line_num-1)
                for sub_line_id in range(sub_line_num):
                    ratio = sub_line_id * step
                    pos1 = (end_pos_up[0]*ratio+begin_pos[0]*(1.0-ratio), end_pos_up[1]*ratio+begin_pos[1]*(1.0-ratio))
                    pos2 = (end_pos_down[0]*ratio+begin_pos[0]*(1.0-ratio), end_pos_down[1]*ratio+begin_pos[1]*(1.0-ratio))
                    cv2.line(out_img, (int(pos1[0]), int(pos1[1])), (int(pos2[0]), int(pos2[1])), (0, 0, 255), 3)
                # contours = np.array([contours,]).astype("int32")
                # cv2.drawContours(out_img, contours, -1, color=(0,255,0), thickness=1)
            elif lastBond.m_type == "~/":
                cv2.line(out_img, (int(begin_pos[0]), int(begin_pos[1])), (int(end_pos[0]), int(end_pos[1])), (255, 128, 0), 1)
            elif "@" in lastBond.m_type or "#" in lastBond.m_type:
                # superbond 
                tgt_angle = lastBond.m_angle % 360
                if tgt_angle >= 180:
                    tgt_angle = tgt_angle - 180
                offset_x = shift * math.sin(tgt_angle * math.pi / 180.0)
                offset_y = shift * math.cos(tgt_angle * math.pi / 180.0)
                cv2.line(out_img, (int(begin_pos[0] + offset_x), int(begin_pos[1] + offset_y)), (int(end_pos[0] + offset_x), int(end_pos[1] + offset_y)), (0, 0, 255), 1)
                cv2.line(out_img, (int(begin_pos[0] - offset_x), int(begin_pos[1] - offset_y)), (int(end_pos[0] - offset_x), int(end_pos[1] - offset_y)), (0, 0, 255), 1)
                cv2.line(out_img, (int(begin_pos[0]), int(begin_pos[1])), (int(end_pos[0]), int(end_pos[1])), (255, 0, 0), 1)
            else:
                raise NotImplementedError("can not rend bond type = %s" % lastBond.m_type)
        if not extend:
            continue
        for bond in curAtom.out_bonds:
            atom_stack.append((bond.end_atom, bond))
        for bond in curAtom.in_bonds:
            atom_stack.append((bond.begin_atom, bond))
    
    # draw atoms next
    atom_stack = [(rootAtom, None)]
    visited_atoms = []
    while len(atom_stack):
        curAtom, lastBond = atom_stack.pop()
        if curAtom is None:
            continue
        if curAtom in visited_atoms:
            continue
        visited_atoms.append(curAtom)
        tgt_x = curAtom.pos_x
        tgt_y = curAtom.pos_y
        
        if rend_name:
            temp_text = curAtom.m_text
            curAtom.m_text = curAtom.name.replace("Atom_", "a")
        if curAtom.m_text is None or curAtom.m_text == "" or curAtom.m_text == "\\circle":
            radius = 1
            if isinstance(curAtom, CircleAtom):
                radius = int(curAtom.m_radius * scale)
            if curAtom.m_text == "\\circle":
                radius = int(scale * 0.5)

            cv2.circle(out_img, (int(tgt_x), int(tgt_y)), radius, (0, 0, 0), 1)
        else:
            normed_text = "".join(curAtom.normed_text())
            size = cv2.getTextSize(normed_text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1 * scale / 150.0, thickness=1)
            ch_width, ch_height = size[0]
            top = int(tgt_y - ch_height/2.0)
            bottom = int(tgt_y + ch_height/2.0)
            left = int(tgt_x - ch_width/2.0)
            right = int(tgt_x + ch_width/2.0)
            cv2.rectangle(out_img, (left, top), (right, bottom), (255, 255, 255), -1)
            cv2.rectangle(out_img, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.putText(out_img, normed_text, (left, bottom), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1 * scale / 150.0, color=(0, 0, 0), thickness=1)
            pass
        if rend_name:
            curAtom.m_text = temp_text
        for bond in curAtom.out_bonds:
            atom_stack.append((bond.end_atom, bond))
        for bond in curAtom.in_bonds:
            atom_stack.append((bond.begin_atom, bond))

    for atom in all_atoms:
        atom.pos_x = atom.pos_x + min_x - margin_w
        atom.pos_y = atom.pos_y + min_y - margin_h
    return out_img

'''输入Atom实现可视化'''
def rend(rootAtom, scale=100, rend_name=0):
    all_atoms = SimulateCoord(rootAtom, scale=scale) # 计算模拟坐标
    img =  rend_atoms(all_atoms, scale, rend_name=rend_name)
    SimulateCoord(rootAtom, 1)
    return img


