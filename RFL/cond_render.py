#INFO: chemfig_cond_render_v3
import os, sys
import cv2
from chemfig_struct import *
from chemfig_parser import *
from chemfig_ops import *
from chemfig_ssml_struct import bond_types
import text_render
import math
import numpy
import pdb
from copy import deepcopy
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
import random

def refine_graph(root_atom:Atom):
    all_atoms = GetAllAtoms(root_atom)
    all_bonds = set()
    for atom in all_atoms:
        text_arr = atom.normed_text()
        atom.__dict__["m_normed_text"] = text_arr
        atom.all_bonds = []
        for bond in atom.out_bonds:
            angle = math.atan2(-bond.end_atom.pos_y + bond.begin_atom.pos_y, bond.end_atom.pos_x - bond.begin_atom.pos_x)
            angle = round(angle*180/math.pi)%360
            atom.all_bonds.append((bond, angle, bond.end_atom))
            all_bonds.add(bond)
        for bond in atom.in_bonds:
            angle = math.atan2( -bond.begin_atom.pos_y + bond.end_atom.pos_y, bond.begin_atom.pos_x - bond.end_atom.pos_x)
            angle = round(angle*180/math.pi)%360
            atom.all_bonds.append((bond, angle, bond.begin_atom))
            all_bonds.add(bond)

    for bond in all_bonds:
        angle = math.atan2(-bond.end_atom.pos_y + bond.begin_atom.pos_y, bond.end_atom.pos_x - bond.begin_atom.pos_x)
        angle = round(angle*180/math.pi)%360
        length = math.sqrt(math.pow(bond.end_atom.pos_y - bond.begin_atom.pos_y, 2) + math.pow(bond.end_atom.pos_x - bond.begin_atom.pos_x, 2))
        bond.m_angle = angle
        bond.m_length = length

#
def random_cond_render(root_atom: Atom):
    out_units = [] #(main_head, cand_head, hook_head, cond_input[place_ind+cand_head])
    all_atoms = GetAllAtoms(root_atom)
    all_atoms = sorted(all_atoms, key=lambda x:x.pos_x*10000-x.pos_y)
    start_atom = all_atoms[0]
    cand_stack = {} #{ key: time_t , value: (start_atom, bond, next_atom, next_angle)}

    cur_bond = None
    cur_angle = None
    cur_atom = start_atom
    cur_cond = None #(time_t, ,)  #time_t of "\place"
    cur_hook = None #(time_t, bond_type)
    visited = {} #{key:atom_name, value:(time_t, bond, angle, atom)} #time_t of "<ea>"  of cur_atom
    while True:
        # debug
        # print("{}\t{}\t{}\t{}".format(cur_atom, cur_angle, cur_bond, cur_cond))
        # if cur_atom.name == "Atom_6":
        #     pdb.set_trace()
       
        # print current bond and atom
        if cur_bond is not None:
            angle = int(15 * ((cur_angle+7.5)//15.0))%360
            bond_str = "{}[:{}]".format(cur_bond.m_type, angle)
            out_units.append([bond_str, None, None, cur_cond])
        for ch in cur_atom.m_normed_text:
            out_units.append([ch, None, None, None])
        # end_unit = ["<ea>", None, None, None]
        # out_units.append(end_unit)
        visited[cur_atom.name] = [len(out_units)-1, cur_bond, cur_angle, cur_atom]

        # statis next cands
        all_bonds = []#(tgt_bond, tgt_angle, tgt_atom)
        end_unit = ["<ea>", None, None, None]
        for tgt_bond, tgt_angle, tgt_atom in cur_atom.all_bonds:
            tgt_angle = int(15 * ((tgt_angle+7.5)//15.0))%360
            if tgt_bond == cur_bond:
                continue
            if tgt_atom.name in visited:
                value = visited[tgt_atom.name]
                cur_t = value[0] + 1
                key_angle = int(15 * ((tgt_angle+180+7.5)//15.0))%360
                match_time_t = None
                while out_units[cur_t][0].find("\\place") != -1:
                    if out_units[cur_t][0] == "\\place[:{}]".format(key_angle) and cur_t in cand_stack:
                        match_time_t = cur_t
                    cur_t += 1
                if match_time_t is None:
                    # print("----------------------------")
                    # for unit_id, unit in enumerate(out_units):
                    #     print("{:>2}\t{}".format(unit_id, unit))
                    # print("----------------------------")
                    # pdb.set_trace()
                    raise ValueError("File \"{}\", line {} : hook must in cand_stack, key={}".format(__file__, sys._getframe().f_lineno, (cur_t, key_angle)))
                cand_stack.pop(match_time_t)
                # print("connect key = {}".format(key))
                # tgt_bond.m_type
                if end_unit[2] is None:
                    end_unit[2] = []
                end_unit[2].append([match_time_t, tgt_bond.m_type ]) #(time_t, angle)
                continue
            all_bonds.append((tgt_bond, tgt_angle, tgt_atom))
        if end_unit[2] is not None and len(end_unit[2]) > 0:
            out_units.append(end_unit)
        random.shuffle(all_bonds)
        # calc next point
        if len(all_bonds) == 0:
            out_units.append(["\\eol", None, None, None])
            if len(cand_stack) <= 0:
                break
            keys = list(cand_stack.keys())
            random.shuffle(keys)
            next_key = keys[0]
            start_atom, next_bond, next_atom, next_angle = cand_stack.pop(next_key)
            next_cond = next_key
            pass
        elif len(all_bonds) > 1:
            # place_unit = ["\\place", [], None, None]
            # out_units.append(place_unit)
            # cur_time_t = len(out_units) - 1
            selected_time_t = None
            all_bonds = sorted(all_bonds, key=lambda x:x[1])
            tgt_cand_id = random.randint(0, len(all_bonds) - 1)
            for cand_id, (cand_bond, cand_angle, cand_atom) in enumerate(all_bonds):
                key_angle = int(15 * ((cand_angle+7.5)//15.0))%360
                cur_token = "\\place[:{}]".format(key_angle)
                out_units.append([cur_token, None, None, None])
                cur_time_t = len(out_units) - 1
                if cand_id != tgt_cand_id:
                    cand_stack[cur_time_t] = (cur_atom, cand_bond, cand_atom, key_angle) #{ key:(time_t, angle) , value: (start_atom, bond, next_atom)}
                else:
                    selected_time_t = cur_time_t
            out_units.append(["\\eol", None, None, None])
                
            #print(cand_stack)
            next_bond = all_bonds[tgt_cand_id][0]
            next_angle = all_bonds[tgt_cand_id][1]
            next_atom = next_bond.end_atom if next_bond.begin_atom == cur_atom else next_bond.begin_atom
            next_cond = selected_time_t
            pass
        else: # num bonds == 1
            next_bond = all_bonds[0][0]
            next_angle = all_bonds[0][1]
            next_atom = next_bond.end_atom if next_bond.begin_atom == cur_atom else next_bond.begin_atom
            # pdb.set_trace()
            next_cond = None
            # if next_atom.name in visited:
            #     value = visited[next_atom.name]
            #     key_angle = (next_angle - 180)%360
            #     key = (value[0] + 1, key_angle)
            #     if key not in cand_stack:
            #         print("----------------------------")
            #         for unit_id, unit in enumerate(out_units):
            #             print("{:>2}\t{}".format(unit_id, unit))
            #         print("----------------------------")
            #         pdb.set_trace()
            #         raise ValueError("File \"{}\", line {} : hook must in cand_stack, key={}".format(__file__, sys._getframe().f_lineno, key))
            #     cand_stack.pop(key)
            #     print("connect key = {}".format(key))
            #     end_unit[2] = key #(time_t, angle)
            #     if len(cand_stack) <= 0:
            #         break
            #     keys = list(cand_stack.keys())
            #     random.shuffle(keys)
            #     next_key = keys[0]
            #     start_atom, next_bond, next_atom = cand_stack.pop(next_key)
            #     next_angle = next_key[1]
            #     next_cond = next_key
            #     pass
        # out_units.append(end_unit)
        cur_atom = next_atom
        cur_bond = next_bond
        cur_angle = next_angle
        cur_cond = next_cond
        cur_hook = None

        pass
    # print("----------------------------")
    # for unit_id, unit in enumerate(out_units):
    #     print("{:>2}\t{}".format(unit_id, unit))
    # print("----------------------------")
    # pdb.set_trace()
    return out_units

def chemfig_random_cond_parse(inputStr, echo=False, bond_dict=[], atom_dict=[], preprocess=True):
    # 获取inputStr中所有bond的索引
    ori_inputstr = deepcopy(inputStr)
    pre_bond_list = [] # [bond, index]
    pre_atom_list = [] # [atom, index]
    cur_bond_types = deepcopy(bond_types)
    remove_list = ['-']
    for item in remove_list:
        cur_bond_types.remove(item)
    for i, item in enumerate(inputStr.split(' ')):
        if '[:' in item and item.endswith(']'):
            pre_bond_list.append([item, i])
        elif item.startswith('?[') and item.endswith(']') and ',' in item: # end hook
            pre_bond_list.append([item, i])
        elif item in cur_bond_types:
            pre_bond_list.append([item, i])
    for i, item in enumerate(inputStr.split(' ')):
        if item == '\Superatom':
            pre_atom_list.append([item, i])

    if echo:
        print(inputStr)

    if echo:
        print("--------------------------1,处理转义------------------------------")
    inputStr = process_escape_strs(inputStr)

    if echo:
        print(inputStr)

    if echo:
        print("--------------------------2,拆分单元-------------------------------")
    units = parse_units(inputStr) # parse无法解析 * 标记
    # 处理branch_info和ring_branch_info
    # branch_units = []
    # if branch_info is not None:
    #     temp_list = [item for item in branch_info if item is not None]
    #     for unit in units:
    #         if unit[1] == 'bond':
    #             branch_units.append(temp_list.pop(0))
    #         else:
    #             branch_units.append(None)
    #     assert len(branch_units) == len(units), "chemfig->图过程中branch_info信息有误."

    if echo:
        print(units)
        print("--------------------------3,解析原子-------------------------------")
    
    elements = parse_element(units, bond_dict=bond_dict, atom_dict=atom_dict) # 需要在这里为bond加上额外的分支位置信息, 为atom加上ring_bond信息
    
    if echo:
        for ele in elements:
            print(ele)
    
    if echo:
        print("--------------------------4,分析分支结构-------------------------")
    root_atom, root_branch = connect_relation(elements, echo=echo)
    #simulate_coord(root_atom, scale=1, remove_dup=1, norm_circle=1, connect_distant=1)
    all_atoms = text_render.pre_process(root_atom, preprocess=False, bond_dict=bond_dict) # 这里删除掉了一个化学键
    # all_atoms = remove_self_ring(all_atoms)
    root_atom = all_atoms[0]
    refine_graph(root_atom)
    
    # 将bond_dict的value转化到原始字符串index中
    for index, (bond, pre_bond) in enumerate(zip(bond_dict, pre_bond_list)):
        bond_str = pre_bond[0]
        bond = bond[0]
        if '[:' in bond_str and ']' in bond_str:
            index1 = bond_str.index('[')
            index2 = bond_str.index(']')
            angle_str = bond_str[index1+2: index2]
            if ',' in angle_str:
                angle_str = angle_str[ : angle_str.index(',')]
            if '.' in angle_str:
                angle_str = angle_str[ : angle_str.index('.')]
            
            if bond.m_type == bond_str[0:index1] and str(bond.m_angle) == angle_str:
                bond_dict[index][1] = pre_bond[1] # modify bond_dict index
            else:
                raise ValueError("bond类型不匹配")
        elif bond_str in cur_bond_types:
            if bond.m_type == bond_str:
                bond_dict[index][1] = pre_bond[1] # modify bond_dict index
            else:
                raise ValueError("bond类型不匹配")
        else: # ?[a,{-}]
            index1 = bond_str.index('{')
            index2 = bond_str.index('}')
            if bond.m_type == bond_str[index1+1:index2]:
                bond_dict[index][1] = pre_bond[1] # modify bond_dict index
            else:
                raise ValueError("bond类型不匹配")
    assert len(pre_bond_list) == len(bond_dict), "pre_bond_list和bond_dict长度不相同"
    
    # 将atom_dict的value转化到原始字符串index中
    for index, (atom, pre_atom) in enumerate(zip(atom_dict, pre_atom_list)): # TODO:这里的顺序是否一致?
        atom_dict[index][1] = pre_atom[1]

    if echo:
        from image_render import rend
        cv2.imwrite("rend.jpg", rend(root_atom, scale=150))
        cv2.imwrite("rend_name.jpg", rend(root_atom, scale=150, rend_name=1))
        outArr = random_cond_render(root_atom)
        for ind, rend_unit in enumerate(outArr):
            print("{}\t{}".format(ind, rend_unit))
        pdb.set_trace()
    
    
    return root_atom

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


def process_text_rnd_cond_render(input_arr):
    out_units = []
    for ind, element in enumerate(input_arr):
        if not isinstance(element, Atom):
            out_units.append([element, None, None, None])
            continue
        cur_units = random_cond_render(element)
        out_units += [["\\chemfig", None, None, None], ["{", None, None, None]]
        start_ind = len(out_units)
        for unit in cur_units:
            if isinstance(unit[2], list):
                for hook in unit[2]: #hook [time_t, angle]
                    hook[0] += start_ind
            if unit[3] is not None:
                unit[3] += start_ind
            out_units.append(unit)
        out_units.append(["}", None, None, None])
    return out_units

    

if __name__ == "__main__":
    str1 = "CH_{3}-CH(-[2]CH_{3}) -CH_{3}"
    str2 = "H- N(-[2]H)(-[6]H) - N(-[2]H)(-[6]H)-H"
    str3 = "**[30, 150]6(-(-[6]OH)-(-[7]OCH_{3})--(-[2]CH=[0]CHNO_{2})--)"
    str4 = "**[90, 190]6(-(-[6]OH)-(-[7]OCH_{3})--(-[2]CHCH_{2}NO_{2}-[4]HO)--)"
    str5 = "C(-[0,0.5]H)(-[4]H_{3}C)(-[1]C(=[2]O)(-[0]O(-[7]C(-[0]H)(-[4,0.5]H_{3}C)(-[5]C(=[6]O)(-[4]O(-[3,0.8]))))))"
    str6 = "H_{3}C-C(-[6]CH_{2}(-[6]C(-[0]O-[0]CH_{3})(=[6]O)(-[4]H_{3}C)))(=[2]O)-O-CH_{3}"
    str7 = "[:45]*6(=(-[7]R)-=-(-[4]{(}CH_{2}{)}_{3}-[4]H_{2}N)=-)" #error
    str8 = "**6((-[:190]O-[3]-[1]O)- - -(-[1](-[7](=[1]O)(-[5]N**5([:-90]-----))))--(-[:170])-)"
    str9 = "H \Charge{180=\:,90=\:,0=\:,-90=\:,90:6pt=H,-90:6pt=H}{N} H"
    str10 = "**[90, 190]6(-(-[6]OH)-(-[7]OCH_{3})--([10]-CHCH_{2}NO_{2}-[4]HO)--)"

    str11 = "**6(---(-[1](-[2]COCl)-[7]-[1](-[2]OH)-[7]**6([:-30]------))---)"
    str12 = "**6(--*6(---(-[:30]**6(--=-=-))-(=[2]O)--)----)"  #ring in ring
    str13 = "CH_{3}-[:-45]CH([:45]-[::90]CH_{3}-CH5) -CH_{3}"

    str14 = "[:30]---[:-30](--([::30]-C-H*6(---=-=))-)-[:-50]--[0]*6([::30]------)"
    str15 = "O-[5]C(=[4]O)-[6]CH(-[4]CH_{3})-[7]O-[1]C(=[0]O)-[2]CH-[3]"
    str16 = "[:30]*6(=-(-[0])-=-(-[4](-[3])(-[5]))-)"
    str17 = "CH_{2}OH(-[6]CHOH(-[6]CH_{2}OH))"
    str18 = "H\\Charge{180=\\:,90=\\:,0=\\:,-90=\\:,90:6pt=H,-90:6pt=H}{C}\\Charge{90=\\:,0=\\:,-90=\\:,90:6pt=H,-90:6pt=H}{C}H"
    str19 = "H_{2}B(-[6,,2,2]H_{2}N?[a])-NH_{2}(-[6]B?[a]H_{2})"
    str20 = "N?[a]H-[1]-[2](-[0]CH_{3})-[3]O-[5]-[6]?[a]"

    str21 = "**6(---(-[0](*6([:330]---=--))(-[,2]CHO))---)" #multiple in-bonds
    str22 = "*6(--(=CH2)----)"
    str23 = "*6(=-=(-[:30]NO_{2})-=-)"
    str24 = "*6((-[:230]Br)-(-[:-90]Br)-(-Br)-(-Br)---)"

    str25 = "*6(CH-CH-CH-CH-CH-CH-)" #virtual bond #苯环开头不能是原子
    str26 = "C(-[3]OHC)(-[5]C)=C(-[1]C{(CH_{3})}_{3})(-[7]CHO)" #virtual bond2 #括号显示问题,为显示真正的括号
    str27 = "**6(---(-[0](NO_{2}))---)"#多一对括号
    str28 = "**6(---(-[0](Br))---)"#多一对括号
    str29 = "H\Charge{180=\: ,90=\: ,-90=\: ,0=\ :}{O}\Charge{0=\:,90=\:,-90=\:}{O}H" #电子云虚拟键
    str30 = "[:270]**6(--(-Br)----)"
    str31 = "[:0]**6(--(-Br)----)"
    str32 = "**6(--(-[0]Br)----)"

    str33 = "[:30]**6(--(-COOH)---(-HO)-)"
    str34="[:30]*6(=-(-[0]COOH)=-=(-[4]HO)-)"

    str35 = "A?[a]-B(-[1]W?[a,2]-X?[b])(<|[7]Y-Z?[b,1])-C?[b,{>}]"
    str36 = "**5(-----)"
    
    str37 = "\Charge{180=\: }{Si} \Chembelow{\Chemabove{\Charge{180=\:,90=\:,0=\:,-90=\:}{H}}{\Charge{180=\: ,90=\: ,0=\ :}{Cl}}}"
    str38 = "CH_{2}(-[4,1.5]CH_{2})(-[3]CH_{2}(-[5,,1]))"
    str39 = "[:15]C*5(-C(=[:330]O)-O-C(=[:80]O)-C=)"
    str40 = "H \Charge{180=\:,90=\:,0=\:,-90=\:,90:6pt=H,-90:6pt=H}{C} Cl"

    str41 = "C-[1]-[-1]**6(---**6(----(-Cl)-)---)"
    str42 = "**6(--**6(----**6(----)-)----)"
    str43 = "*6((-Cl)=(\\xcancel{})-*6(-N(-[6])-(=[7]O)--O-)=-(-(=[3]O)-[1]O-[7]Na)=-)" #分支第一个元素为原子

    str44 = "A-=-C(C-[-2]=-)-[2]--B"
    str45 = "**6((-Cl)--(*6(-N=(-[7]=[1]-[7](-[:-30,0]**6(----(-[1](-[2]Br)-[7]-[1]-[7](-[:-30,0]**6(-(-[5](-[6])=[3]O)-----)))--)))-=-))----)"
    str46 = "**6((-Cl)--(*6(-N=(-[7]=[1]-[7](-[:-30,0]**6(----(-[1](-[:80]S(-[:60,1.2]-[:-60](-[:-90,0.01]*3(---))-[:60]-[7]COOH))-[7]-[1]-[7](-[:-30,0]**6(-(-[5]C(-[6])=[3]O)-----)))--)))-=-))----)"
    str47 = "CH_{3}-C(-[6]*6([6,0.5,]=[6,0.5,]-[6,0.5,]=[6,0.5,]-[6,0.5,]=[6,0.5,]-))(-[2]OH)-C(-[6]*6([6,0.5,]=[6,0.5,]-[6,0.5,]=[6,0.5,]-[6,0.5,]=[6,0.5,]-))(-[2]H)-CH_{3}" #环内的键指定了方向

    str48 = "*6((-[5]H_{3}CO)=- ((*5(--(-[0]OH)-(=[1]O)--)))=-=(-[3]H_{3}CO)-)" # 环在共享边的时候，键型不一致，导致合并不了，待解决
    str49 = "**6((-HO)--(-[7]O-[1,1.7,,])-(-[1]O-[7,1.7,,])---)" # 长度误差带来的错误闭合
    str50 = "**6((-O-[4]C(=[2]O)-[4]CH_{3})--(*5(-(-[7])=-O-))---(-)-)" #括号包环中环
    str51 = "**6(---(-[1](=[2]O)-[7]O-[1]-[7]-[1]O-[7](=[6]O)-[1]([:30]**6(------)))---)"#括号包环中环
    str52 = "[:35]*5((-[5])-(*5(-(-[7])-(=[0]O)-(-[1])--))--(-[3])-(=[4]O)-)"#括号包环中环

    str53 = "**6(--(-*6([:0]-=-=-=))----)"
    str54 = "*6(=-(*6(-=-(-[0]CH_{3})=--))=-=-)"
    str55 = "?[a] -[:90] =[:30] -[:330] ?[b] ( -[:30] =[:330] ( -[:0] C H _ { 3 } ) -[:270] =[:210] -[:150] ?[b,{-}] -[:210] ?[a,{=}]"

    str56 = "**6(([:144]*5(--O--O-))---(-[1]CHO)---)"
    str57 = "H-C(=[2]O\cdots H?[a])-O-H\cdots O=[2,,2]C(-[0]H)([4]-[4]O?[a])"
    str58 = "\\Charge{ 0=\\.,180 = \:,90=\: }{ H} \\Charge{0=\\:}{\\boxed{NCH}}"
    str59 = "*6(=-(-[0]OH)=(-[0]CH_{3})-=-)"
    str60 = "CH_{3}CH_{2}C(-[6]C?[a](=[6]O))(=[2,,5]O)(-[6,,5]C?[a]H(-[0]CH_{3}))"
    str61 = "?[a] ( -:[:30] \circle ) -[:90] -[:30] ( -[:45] * ) ( -[:90] C O O H ) -[:330] ( -[:0] N H _ { 2 } ) ( -[:45] * ) -[:270] -[:210] ?[a,{-}]"
    inputStr = str61
    atom = chemfig_random_cond_parse(str60, True)
    # pdb.set_trace()
    pass

