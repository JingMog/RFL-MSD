import os, sys
import re
import pdb
import numpy as np
from chemfig_struct import *
import cv2
import pickle

replace_dict = {
    "\\equiv":"~",
    "\\mcfabove":"\\Chemabove",
    "\\mcfbelow":"\\Chembelow", 
    #"\\mcfminus":"\-",
    #"\\mcfplus":"+"
}

def process_virtual_bond(inArr): # used only for old rec format
    outArr = []
    ind = 0
    while ind < len(inArr):
        cur_ch = inArr[ind]
        if cur_ch == "-":
            if ind < len(inArr) -1 and inArr[ind+1] == "-":
                outArr.append("-:")
                ind += 2
                continue
        outArr.append(cur_ch)
        ind += 1
    return "".join(outArr)

def process_replace(inputStr):
    outStr = inputStr
    for key in replace_dict:
        outStr = outStr.replace(key, replace_dict[key])
    return outStr

def split_bracket_pairs(inputStr:list):
    out_groups = []
    curLevel = 0
    cur_begin = None
    for i, ch in enumerate(inputStr):
        if ch == "{":
            if cur_begin is None:
                cur_begin = i
            curLevel += 1
        elif ch == "}":
            curLevel -= 1
            if curLevel <= 0 and cur_begin is not None:
                out_groups.append((cur_begin, i+1))
                cur_begin = None
    out_arr = []
    length = len(inputStr)
    flag_arr = np.zeros((length))
    cur_index = 0
    for begin, end in out_groups:
        out_arr += inputStr[cur_index:begin]
        #out_arr.append("".join(inputStr[begin:end]))
        out_arr += inputStr[begin:end]
        flag_arr[begin:end] = 1
        cur_index = end
    out_arr += inputStr[cur_index:]
    #out_arr = list(filter(lambda x:x!="", out_arr))
    return out_arr, flag_arr
         


 
def process_escape_strs(inputStr, is_process_virtual_bond=False):
    inputStr = process_replace(inputStr)
    if is_process_virtual_bond is True:
        inputStr = process_virtual_bond(inputStr)

    #inputStr, rep_dict = split_bracket_pairs(inputStr)
    #pdb.set_trace()

    outStr = []
    groups = []
    used = np.zeros(len(inputStr))
    for escape_str in escape_strs + bond_types:
        for match_obj in re.finditer(re.escape(escape_str), inputStr):
            begin, end = match_obj.span()
            if sum(used[begin:end]) > 0:
                continue
            if end - begin > 0:
                groups.append((begin, end))
                used[begin:end] = 1
        
    groups = sorted(groups, key=lambda x: x[0])
    # for begin, end in groups:
    #     print("({}, {})".format(begin, end))
    #     print(inputStr[begin:end])
    last_end = 0
    for begin, end in groups:
        assert begin >= last_end
        outStr += list(inputStr[last_end:begin])
        outStr.append(inputStr[begin:end])
        last_end = end
    if last_end < len(inputStr):
        outStr += list(inputStr[last_end:])

    #pdb.set_trace()
    # # process longstr bond
    # for bond_type 

    outStr, flag_arr = split_bracket_pairs(outStr)

    # outStr = process_virtual_bond(outStr)
    # outStr = process_replace(outStr)
    return outStr, flag_arr

def parse_units(inputStr):
    inputStr, flag_arr = inputStr
    def list_find(in_arr, pattern, left=0, right=None):
        if right is None:
            right = len(in_arr)
        try:
            return in_arr.index(pattern, left, right)
        except:
            return -1
    
    def find_bracket(in_arr, sos="{", eos="}", num=1, left=0, right=None):
        pairs = []
        if right is None:
            right = len(in_arr)
        cur_ind = left
        curLevel = 0
        begin = None
        end = None
        while True:
            if in_arr[cur_ind] == sos:
                if curLevel == 0 and begin is None:
                    begin = cur_ind
                curLevel += 1
            elif inputStr[cur_ind] == eos:
                curLevel -= 1
            if curLevel == 0 and begin is not None:
                end = cur_ind + 1
                pairs.append((begin, end))
                begin = None
            if len(pairs) >= num:
                break
            cur_ind += 1
            if cur_ind >= len(in_arr):
                break
        return pairs

    ind = 0
    units = []
    types = []
    cur_state = "text"
    cur_state = "begin"
    last_state = "begin"
    while True:
        if inputStr[ind] in bond_types and flag_arr[ind] == 0:
            cur_state = "bond"
            units.append(inputStr[ind])
            types.append(cur_state)
            ind += 1
        elif inputStr[ind] in special_chars and flag_arr[ind] == 0:
            cur_state = "special"
            if inputStr[ind] == "[":
                #right_pair_pos = inputStr.index("]", ind)
                right_pair_pos = list_find(inputStr, "]", ind)
                if right_pair_pos == -1:
                    raise ValueError("can not find ] for [ in pos %d of input %s" % (ind, inputStr))
                units.append("".join(inputStr[ind:right_pair_pos + 1]))
                types.append(cur_state)
                ind = right_pair_pos + 1
            elif inputStr[ind] == "(" or inputStr[ind] == ")":
                units.append(inputStr[ind])
                types.append(cur_state)
                ind += 1
            elif inputStr[ind] == "\\Charge":
                units.append(inputStr[ind])
                types.append("special")
                begin = list_find(inputStr, "{", ind)
                # assert inputStr[ind+1] == "{"
                assert list(filter(lambda x:x != " ", inputStr[ind+1:begin])).__len__() == 0
                assert begin != -1
                curLevel = 1
                #cur_ind = ind + 2
                #begin = ind+1
                cur_ind = begin + 1
                pairs = []
                while True:
                    if inputStr[cur_ind] == "{":
                        if curLevel == 0 and begin is None:
                            begin = cur_ind
                        curLevel += 1
                    elif inputStr[cur_ind] == "}":
                        curLevel -= 1
                    if curLevel == 0 and begin is not None:
                        end = cur_ind + 1
                        # units.append(inputStr[begin:end])
                        # types.append("special")
                        pairs.append((begin, end))
                        begin = None
                    if len(pairs) >= 2:
                        break
                    cur_ind += 1
                for ii, (begin, end) in enumerate(pairs):
                    units.append("".join(inputStr[begin:end]))
                    types.append("special")
                    if ii > 0:
                        # assert begin == pairs[ii-1][1]
                        assert list(filter(lambda x:x != " ", inputStr[pairs[ii-1][1]:begin])).__len__() == 0
                ind = pairs[-1][-1]
            elif inputStr[ind] == "*":
                # **[angle1, angle2]6(------)
                if ind + 1 < len(inputStr) and inputStr[ind + 1] == "*":
                    units.append("**")
                    ind += 2
                else:
                    units.append("*")
                    ind += 1
                types.append(cur_state)
                right_pos = list_find(inputStr, "(", ind)
                left_bracket_pos = list_find(inputStr, "[", ind, right_pos)
                right_bracket_pos = list_find(inputStr, "]", ind, right_pos)
                if left_bracket_pos == -1:
                    assert right_bracket_pos == -1 and right_pos >= ind + 1
                    for i in range(ind, right_pos):
                        assert len(inputStr[i]) == 1 and ((ord(inputStr[i]) >= ord("0") and ord(inputStr[i]) <= ord("9")) or inputStr[i]==" ")
                    units.append("".join(inputStr[ind:right_pos]).replace(" ", ""))
                    types.append("special")
                else:
                    assert right_bracket_pos != -1 and right_bracket_pos > left_bracket_pos
                    # units.append(inputStr[left_bracket_pos, right_bracket_pos+1])
                    # types.append("special")
                    assert right_pos >= right_bracket_pos + 1
                    for i in range(right_bracket_pos + 1, right_pos):
                        assert len(inputStr[i]) == 1 and ((ord(inputStr[i]) >= ord("0") and ord(inputStr[i]) <= ord("9")) or inputStr[i]==" ")
                    units.append("".join(inputStr[left_bracket_pos:right_pos]).replace(" ", ""))
                    types.append("special")
                ind = right_pos
            elif inputStr[ind] == "?":
                assert inputStr[ind+1]=="["
                begin = ind + 1
                end = list_find(inputStr, "]", ind+2)
                if end == -1:
                    raise ValueError("can not find ] for [ in pos %d of input %s" % (ind+1, inputStr))
                units.append("".join(inputStr[ind:end+1]))
                types.append("special")
                ind = end+1
            elif inputStr[ind] == "\\mcfcringle":
                # pdb.set_trace()
                pairs = find_bracket(inputStr, "{", "}", 1, ind+1)
                if len(pairs) < 1:
                    warnings.warn("can not find bracket pair for mcfcringle, pos {} at {}".format(ind, " ".join(inputStr)))
                    ind += 1
                    continue
                left, right = pairs[0]
                #assert inputStr[ind+1:left].replace(" ", "") == ""
                assert list(filter(lambda x:x != " ", inputStr[ind+1:left])).__len__() == 0

                units.append(inputStr[ind])
                types.append("special")

                units.append("".join(inputStr[left:right]))
                types.append("special")

                ind = right
            elif inputStr[ind] in ["\\Chembelow", "\\Chemabove", "\\chembelow", "\\chemabove"]:
                pairs = find_bracket(inputStr, "{", "}", 2, ind+1)
                if len(pairs) < 2:
                    warnings.warn("can not find bracket pair for {}, pos {} at {}".format(inputStr[ind], ind, " ".join(inputStr)))
                    ind += 1
                    continue
                _left = ind+1
                for pair in pairs:
                    # assert inputStr[_left:pair[0]].replace(" ", "") == ""
                    assert list(filter(lambda x:x != " ", inputStr[_left:pair[0]])).__len__() == 0
                    _left = pair[1]
                units.append(inputStr[ind])
                types.append("text")
                for pair in pairs:
                    _l, _r = pair
                    units.append(" "+"".join(inputStr[_l:_r])+" ")
                    types.append("text")
                    ind = _r
                cur_state = "text"
                pass
            else:
                raise NotImplementedError("can't handle special for %s" % inputStr[ind])
                ind += 1
        elif inputStr[ind] == " " or inputStr[ind] == "\u00A0" or inputStr[ind] == "\u3000":
            if cur_state == "text": #2022.03.09 we can not remove space in text
                units[-1] += inputStr[ind]
            ind += 1
        else:  #normal text
            cur_state = "text"
            if last_state != "text":
                units.append(inputStr[ind])
                types.append(cur_state)
            else:
                units[-1] += inputStr[ind]
            ind += 1
        last_state = cur_state
        if ind >= len(inputStr):
            break
    assert len(units) == len(types)
    units = list(zip(units, types))
    return units

def parse_element(units, bond_dict=[], atom_dict=[]):
    out_units = []
    ind = 0
    hook_dict = {}
    while True:
        cur_text, cur_type = units[ind]
        # if cur_type == 'bond' and branch_units:
        #     cur_branch_info = branch_units[ind]
        # else:
        #     cur_branch_info = []
        if cur_type == "text":
            atom = Atom(cur_text.strip())
            out_units.append(atom)
            # if ind + 1 < len(units) and units[ind + 1][0][0] == "?":
            #     raise NotImplementedError("not support parse for ?")
            cur_ind = ind + 1
            while cur_ind < len(units):
                if isinstance(units[cur_ind][0], str) and units[cur_ind][0].startswith("?[") and units[cur_ind][0].endswith("]"):
                    hook = DistantHook(units[cur_ind][0])
                    hook_name = hook.m_hookname
                    if hook_name not in hook_dict:
                        hook_dict[hook_name] = [hook]
                        #atom.end_hooks.append(hook)
                        atom.start_hooks.append(hook) # fixed hook direction @2022.02.08
                    else:
                        hook_dict[hook_name].append(hook)
                        # atom.start_hooks.append(hook)
                        atom.end_hooks.append(hook) # fixed hook direction @2022.02.08
                        bond_dict.append([hook, len(out_units)-1])
                elif units[cur_ind][1] == "text": #process cond like "N?[a]H"
                    atom.m_text += units[cur_ind][0]
                else:
                    break
                cur_ind += 1
            ind = cur_ind - 1
            if '\Superatom' in atom.m_text:
                atom_dict.append([atom, len(out_units)-1])
        elif cur_type == "bond":
            bond = Bond(cur_text)
            # bond.branch_info = cur_branch_info
            if ind + 1 < len(units) and units[ind + 1][1] == "special" and units[ind + 1][0][0] == "[" and units[ind + 1][0][-1] == "]":
                bond.parse_attr_from_str(units[ind + 1][0])
                ind += 1
            out_units.append(bond)
            # bond_dict[bond] = len(out_units) - 1 # 保存到bond_dict
            bond_dict.append([bond, len(out_units)-1])
        elif cur_text == "**" or cur_text == "*":
            ring_head = RingHead(cur_text)
            ring_head.parse_attr_from_str(units[ind + 1][0])
            out_units.append(ring_head)
            ind += 1
        elif cur_text.startswith("[") and cur_text.endswith("]"): # 
            out_units.append(AttrStr(cur_text))
        elif cur_text.startswith("?[") and cur_text.endswith("]"): # reconn
            atom = Atom("")
            out_units.append(atom)
            cur_ind = ind
            while cur_ind < len(units):
                if isinstance(units[cur_ind][0], str) and units[cur_ind][0].startswith("?[") and units[cur_ind][0].endswith("]"):
                    hook = DistantHook(units[cur_ind][0])
                    hook_name = hook.m_hookname
                    if hook_name not in hook_dict:
                        hook_dict[hook_name] = [hook]
                        #atom.end_hooks.append(hook)
                        atom.start_hooks.append(hook)
                    else:
                        hook_dict[hook_name].append(hook)
                        #atom.start_hooks.append(hook)
                        atom.end_hooks.append(hook)
                        # bond_dict[hook] = len(out_units) - 1
                        bond_dict.append([hook, len(out_units)-1])
                elif units[cur_ind][1] == "text":
                    atom.m_text += units[cur_ind][0]
                else:
                    break
                cur_ind += 1
            ind = cur_ind - 1
        elif cur_text in ["\\Charge"]:
            charge_atom = Charge("")
            charge_atom.parse_attr_from_str(units[ind + 1][0],units[ind + 2][0])
            out_units.append(charge_atom)
            ind += 2
        elif cur_text == "\\mcfcringle":
            # pdb.set_trace()
            circleAtom = CircleAtom()
            radius_str = units[ind+1][0]
            radius_str = radius_str.strip()[1:-1]
            radius = float(radius_str)
            circleAtom.m_radius = radius
            out_units.append(circleAtom)
            ind += 1
        else:
            out_units.append(cur_text)
            pass
        ind += 1
        if ind >= len(units):
            break
    return out_units

def printBranch(curBranch, curLevel=0):
    #print(curBranch)
    prefix = ""
    for i in range(curLevel):
        prefix += "\t"
    for child in curBranch.childs:
        print("{}{}".format(prefix, child))
        if isinstance(child, Branch):
            printBranch(child, curLevel + 1)
    pass

def connect_relation(elements, echo=False):
    rootBranch = Branch(None)
    branch_stack = [rootBranch]
    curBranch = rootBranch

    curLevel = 0
    level_atom_dict = {curLevel: None}
    for i, element in enumerate(elements):
        if isinstance(element, Atom):
            curBranch.childs.append(element)
            level_atom_dict[curLevel] = element
            pass
        elif isinstance(element, Bond):  #need atom
            if level_atom_dict[curLevel] is None:
                new_null_atom = Atom("")
                level_atom_dict[curLevel] = new_null_atom
                curBranch.childs.append(new_null_atom)
            curBranch.childs.append(element)
            level_atom_dict[curLevel] = None
            pass
        elif isinstance(element, RingHead):
            assert elements[i + 1] == "("
            pass
        elif element == "(":
            #pdb.set_trace()
            if level_atom_dict[curLevel] is None:
                new_null_atom = Atom("")
                level_atom_dict[curLevel] = new_null_atom
                curBranch.childs.append(new_null_atom)
            if i > 0 and type(elements[i - 1]) is RingHead:
                new_branch = Ring(level_atom_dict[curLevel], elements[i - 1])
            else:
                new_branch = Branch(level_atom_dict[curLevel])
            curBranch.childs.append(new_branch)
            branch_stack.append(curBranch)
            curBranch = new_branch
            level_atom_dict[curLevel + 1] = level_atom_dict[curLevel]
            curLevel += 1
            pass
        elif element == ")":
            curBranch = branch_stack.pop()
            level_atom_dict[curLevel] = None
            curLevel -= 1
            pass
        elif isinstance(element, AttrStr):
            curBranch.parse_attr_from_str(element.attr_str)
        else:
            raise NotImplementedError("no implement for element {}".format(element))
            continue
    
    # if echo is True:
    #     print("--------------------------after add------------------------------")
    #     for element in elements:
    #         print(element)
    #     print("--------------------------end after add------------------------------")
    init_parent_info=[{"is_ring":0, "m_absolute_angle":0, "pos_arr":[0], "m_base_angle":None, "last_bond":None}]
    init_shared_info={"is_bond_solved": False, "is_root":True}
    root_atom = rootBranch.build_child_graph(parentInfo=init_parent_info, shared_info=init_shared_info)

    return root_atom, rootBranch

def chemfig_parse(inputStr, echo=False):
    if echo:
        print(inputStr)

    if echo:
        print("--------------------------1,处理转义------------------------------")
    inputStr = process_escape_strs(inputStr)
    if echo:
        print(inputStr)

    if echo:
        print("--------------------------2,拆分单元-------------------------------")
    units = parse_units(inputStr)
    if echo:
        print(units)

    if echo:
        print("--------------------------3,解析原子-------------------------------")
    elements = parse_element(units)
    if echo:
        for ele in elements:
            print(ele)

    if echo:
        print("--------------------------4,分析分支结构-------------------------")

    root_atom, root_branch = connect_relation(elements, echo=echo)
    
    # outArr = rend_text(root_atom, scale=1)
    if echo:
        print("-----rootBrach {}-----".format(root_branch))
        printBranch(root_branch)
        print("------end----------")
        # img = rend(root_atom, scale=150)
        # cv2.imwrite("rend.jpg", rend(root_atom, scale=150))
        # cv2.imwrite("rend_name.jpg", rend(root_atom, scale=150, rend_name=1))

    # if echo:
    #     print("origin: {}".format("".join(inputStr)))
    #     print("normed text: {}".format(" ".join(outArr)))
    # with open("atom.pkl", "wb") as fout:
    #     pickle.dump(root_atom, fout)
    # pdb.set_trace()
    return root_atom, root_branch
    


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
    str34 = "[:30]*6(=-(-[0]COOH)=-=(-[4]HO)-)"

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

    str58 = "H_{2}N-[1]( [: 3 0]*6(-(-I)=(-COOH)-(-I)=(-COOH)-(-I)=))"
    str59 = "-[:153] -[:93] -[:153]N ( -[:213] -[:273] -[:213] ) -[:93] ( =[:153]O ) -[:33] ( =[:333]O ) -[:93]?[b] -[:39] (     -[:345]?[a] -[:285] ( -[:45,,,,draw=none]\mcfcringle{1.3}     ) -[:345] -[:45] -[:105] -[:165]?[a,{-}]  )     ( -[:165,0.85,,,draw=none]\mcfcringle{1.03} )    -[:111]\mcfabove{N}{H} -[:183]?[c] -[:135]     ( -[:255,,,,draw=none]\mcfcringle{1.3} ) -[:195] -[:255]      ( -[:195]Cl ) -[:315] -[:15]?[b,{-}]?[c,{-}]   "
    inputStr = str55
    
    outArr = chemfig_parse(inputStr, True)
    print("".join(outArr))