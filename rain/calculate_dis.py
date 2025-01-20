import Levenshtein

def cal_edit_ops(str1, str2):
    char_idx_dict = dict()
    for item in str1:
        if item not in char_idx_dict:
            char_idx_dict[item] = chr(len(char_idx_dict)) #转成这样是因为同一字符长度，方便计算
    for item in str2:
        if item not in char_idx_dict:
            char_idx_dict[item] = chr(len(char_idx_dict))
    str1 = ''.join([char_idx_dict[item] for item in str1])
    str2 = ''.join([char_idx_dict[item] for item in str2])
    ops = Levenshtein.editops(str1, str2) #计算如果第一个字符串变成第二个字符串需要哪些操作
    return ops


def count_ops(ops):
    insert_nums = sum([1 for op_name, *_ in ops if op_name=='delete'])
    substitute_nums = sum([1 for op_name, *_ in ops if op_name=='replace'])
    delete_nums = sum([1 for op_name, *_ in ops if op_name=='insert'])
    assert delete_nums + substitute_nums + insert_nums == len(ops)
    return delete_nums, substitute_nums, insert_nums