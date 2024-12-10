from collections import OrderedDict

__post_line_replace_map = OrderedDict((
    ("\\not =", "\\neq"),
    ("\\not \\leq", "\\nleq"),
    ("\\not \\geq", "\\ngeq"),
    ("\\not \\leqq", "\\nleqq"),
    ("\\not \\ngeqq", "\\ngeqq"),
    ("\\not \\cong", "\\ncong"),
    ("\\not \\lt", "\\nless"),
    ("\\not \\gt", "\\ngtr"),
    ("\\not \\leftarrow", "\\nleftarrow"),
    ("\\not \\Leftarrow", "\\nLeftarrow"),
    ("\\not \\rightarrow", "\\nrightarrow"),
    ("\\not \\Rightarrow", "\\nRightarrow"),
    ("\\not \\subset", "\\nsubset"),
    ("\\not \\supset", "\\nsupset"),
    ("\\not \\subseteq", "\\nsubseteq"),
    ("\\not \\supseteq", "\\nsupseteq"),
    ("\\not \\in", "\\notin"),
    ("\\not \\exists", "\\nexists"),
    ("\\not / /", "\\nparallel"),

    ("\\dot { = }", "\\doteq"),
    ("\\dot { + }", "\\dotplus"),

    ("\\boxed { + }",       "\\boxplus"),
    ("\\boxed { - }",       "\\boxminus"),
    ("\\boxed { \\cdot }",  "\\boxdot"),
    ("\\boxed { \\times }", "\\boxtimes"),
    ("\\boxed { }", "\\square"),

    ('\\textcircled { \\times }', '\\otimes'),
    ('\\textcircled { + }',       '\\oplus'),
    ('\\textcircled { \\cdot }',  '\\odot'),
    ('\\textcircled { - }',       '\\ominus'),
    ('\\textcircled { \\div }',   '\\odiv'),
    ("\\textcircled { }", "\\bigcirc"),


    # 五角星
    ('\\ding 7 2', '\\bigstar'),
    ('\\ding { 7 2 }', '\\bigstar'),
    ('\\star', '\\bigstar'),
    ('\\ding 7 3', '\\UC_2606'),
    ('\\ding { 7 3 }', '\\UC_2606'),

    # 解决无法渲染符号的字符替换问题
    ("\\pxsbx", "\\UC_25B1"),
    ("\\pxdy", "\\xlongequal { // }"),
    ("\\oversetfrown", "\\overset { \\frown }"),
    ("\\DeltaH", "\\Delta H"),
    ("\\jump", "\\\\"),
    ("\\enter", "\\\\"),
))

__post_remove_tokens_list = (
    "\\smear", "\\unk"
)

__post_word_replace_map = OrderedDict((
    ("\\beginmatrix", "\\begin{matrix}"),
    ("\\beginBmatrix", "\\begin{Bmatrix}"),
    ("\\beginVmatrix", "\\begin{Vmatrix}"),
    ("\\beginbmatrix", "\\begin{bmatrix}"),
    ("\\beginpmatrix", "\\begin{pmatrix}"),
    ("\\beginvmatrix", "\\begin{vmatrix}"),
    ("\\endmatrix", "\\end{matrix}"),
    ("\\endBmatrix", "\\end{Bmatrix}"),
    ("\\endVmatrix", "\\end{Vmatrix}"),
    ("\\endbmatrix", "\\end{bmatrix}"),
    ("\\endpmatrix", "\\end{pmatrix}"),
    ("\\endvmatrix", "\\end{vmatrix}"),
))

__post_matrix_left_right_map = OrderedDict((
    ("| |", ("| |", "\\Vert", "\\Vert", "Vmatrix")),
    ("|",   ("|",   "|",   "|",  "vmatrix")),
    ("(",   (")",   "(",  ")",   "pmatrix")),
    ("[",   ("]",   "[",  "]",   "bmatrix")),
    ("\{",  ("\}", "\{", "\}",   "Bmatrix")),
))

__post_chemical_equation_map = OrderedDict((
    ("\\leftarrow", "\\xleftarrow"),
    ("\\rightarrow", "\\xrightarrow"),
    ("\\Leftarrow",  "\\xLeftarrow"),
    ("\\Rightarrow", "\\xRightarrow"),
    ("\\leftrightarrow", "\\xleftrightarrow"),
    ("\\Leftrightarrow", "\\xLeftrightarrow"),
    ("\\hookleftarrow", "\\xhookleftarrow"),
    ("\\hookrightarrow", "\\xhookrightarrow"),
    ("\\mapsto", "\\xmapsto"),
    ("\\rightharpoondown", "\\xrightharpoondown"),
    ("\\rightharpoonup", "\\xrightharpoonup"),
    ("\\leftharpoondown", "\\xleftharpoondown"),
    ("\\leftharpoonup", "\\xleftharpoonup"),
    ("\\rightleftharpoons", "\\xrightleftharpoons"),
    ("\\leftrightharpoons", "\\xleftrightharpoons"),
    ("=", "\\xlongequal"),
    ("\\twoheadrightarrow", "\\xtwoheadrightarrow"),
    ("\\twoheadleftarrow", "\\xtwoheadleftarrow"),
    ("\\rightleftarrows", "\\xrightleftarrows"),
    ("\\leftrightarrows", "\\xleftrightarrows"),
))

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

def convert_matrix(transcription, left_right=False, name=""):
    bad_trans = False
    if "\\beginmatrix" not in transcription and "\\endmatrix" not in transcription:
        return transcription, bad_trans
    matrix_map = OrderedDict()
    not_match_list = list()
    words = transcription.split()
    for idx, word in enumerate(words):
        if word == "\\beginmatrix":
            not_match_list.append(idx)
        elif word == "\\endmatrix":
            if len(not_match_list) == 0:
                # print("name: %s, single \\endmatrix: %s" % (name,transcription))
                bad_trans = True
                words[idx] = ""
            else:
                matrix_map[not_match_list.pop()] = idx
    for nm_idx in not_match_list[::-1]:
        matrix_map[nm_idx] = len(words)
        words.append("\\endmatrix")
    
    global __post_matrix_left_right_map

    left_char_dict  = OrderedDict([(k,    v[1]) for k, v in __post_matrix_left_right_map.items()])
    right_char_dict = OrderedDict([(v[0], v[2]) for k, v in __post_matrix_left_right_map.items()])

    for beg_idx, end_idx in matrix_map.items():
        left_char = "."
        if beg_idx >= 1 and words[beg_idx-1] in left_char_dict:
            left_char = words[beg_idx-1]
            if beg_idx >= 2:
                temp_char = " ".join(words[beg_idx-2:beg_idx])
                if temp_char in left_char_dict:
                    left_char = temp_char
                    words[beg_idx-2] = ""
            words[beg_idx-1] = ""
            words[beg_idx] = left_char_dict[left_char]
        right_char = "."
        if end_idx < len(words) - 1 and words[end_idx+1] in right_char_dict:
            right_char = words[end_idx+1]
            if end_idx < len(words) - 2:
                temp_char = " ".join(words[end_idx+1:end_idx+3])
                if temp_char in right_char_dict:
                    right_char = temp_char
                    words[end_idx+2] = ""
            words[end_idx+1] = ""
            words[end_idx] = right_char_dict[right_char]
        if not left_right and left_char in left_char_dict and __post_matrix_left_right_map[left_char][0] == right_char:
            words[beg_idx] = "\\begin" + __post_matrix_left_right_map[left_char][3]
            words[end_idx] = "\\end" + __post_matrix_left_right_map[left_char][3]
        else:
            if left_char != "." or right_char != ".":
                if left_char != '.':
                    left_char = left_char_dict[left_char]
                if right_char != '.':
                    right_char = right_char_dict[right_char]
                words[beg_idx] = " ".join(["\\left", left_char, "\\beginmatrix"])
                words[end_idx] = " ".join(["\\endmatrix", "\\right", right_char])
    words = [word for word in words if word != ""]
    return " ".join(words), bad_trans

def post_norm_brace(transcription, name=""):
    bad_trans = False
    words = transcription.split()
    if "{" not in words and "}" not in words:
        return transcription, bad_trans
    brace_map = OrderedDict()
    not_match_list = list()
    for idx, word in enumerate(words):
        if word == "{":
            not_match_list.append(idx)
        elif word == "}":
            if len(not_match_list) == 0:
                # print("name: %s, single }: %s" % (name,transcription))
                bad_trans = True
                words[idx] = ""
            else:
                brace_map[not_match_list.pop()] = idx
    for nm_idx in not_match_list[::-1]:
        brace_map[nm_idx] = len(words)
        words.append("}")
    words = [word for word in words if word != ""]
    return " ".join(words), bad_trans


def post_line_replace(transcription):
    global __post_line_replace_map
    for k, v in __post_line_replace_map.items():
        transcription = transcription.replace(k, v)
    return transcription

def post_word_replace(transcription):
    global __post_word_replace_map
    words = transcription.split()
    out_words = []
    for word in words:
        if word == "":
            continue
        if word in __post_word_replace_map:
            word = __post_word_replace_map[word]
        out_words.append(word)
    return " ".join(out_words)

def convert_chemical_equation(transcription):
    global __post_chemical_equation_map
    words = transcription.split()
    if '\\underset' not in words and '\\overset' not in words:
        return transcription
    brace_map = OrderedDict()
    set_map = OrderedDict()
    not_match_list = list()
    for idx, word in enumerate(words):
        if word == "{":
            not_match_list.append(idx)
        elif word == "}":
            cur_key = not_match_list.pop()
            brace_map[cur_key] = idx
        elif word in ['\\underset', '\\overset']:
            set_map[idx] = []

    set_map_idx = list(set_map.keys())
    dep_idx_set = set()
    for idx in set_map_idx:
        if idx+1 in brace_map and brace_map[idx+1]+1 in brace_map:
            set_map[idx] = [idx+2, brace_map[idx+1], brace_map[idx+1]+2, brace_map[brace_map[idx+1]+1]]
            if brace_map[idx+1]+2 in set_map:
                dep_idx_set.add(brace_map[idx+1]+2)
        if len(set_map[idx]) == 0:
            set_map.pop(idx)
    # norm
    set_map_rep_map = OrderedDict()
    for idx, (b1s, b1e, b2s, b2e) in set_map.items():
        if idx in dep_idx_set:
            continue
        b1_norm_trans = convert_chemical_equation(' '.join(words[b1s:b1e])).split()
        down_trans = []
        up_trans = []
        mid_trans = []
        if words[idx] == '\\underset' and words[b2s] == '\\overset':
            down_trans = b1_norm_trans
            up_trans = convert_chemical_equation(' '.join(words[set_map[b2s][0]:set_map[b2s][1]])).split()
            mid_trans = convert_chemical_equation(' '.join(words[set_map[b2s][2]:set_map[b2s][3]])).split()
        else:
            mid_trans = convert_chemical_equation(' '.join(words[b2s:b2e])).split()
            if words[idx] == '\\underset':
                down_trans = b1_norm_trans
            else: # words[idx] == '\\overset':
                up_trans = b1_norm_trans
        if len(mid_trans) == 1 and mid_trans[0] in __post_chemical_equation_map:
            cur_rep_trans = [__post_chemical_equation_map[mid_trans[0]]]
            if len(down_trans) > 0:
                cur_rep_trans.extend(['['] + down_trans + [']']) 
            cur_rep_trans.extend(['{'] + up_trans + ['}'])
            set_map_rep_map[idx] = cur_rep_trans
        else:
            cur_rep_trans = []
            if len(down_trans) > 0:
                cur_rep_trans.extend(['\\underset', '{'] + down_trans + ['}', '{'])
            if len(up_trans) > 0:
                cur_rep_trans.extend(['\\overset', '{'] + up_trans + ['}', '{'])
            cur_rep_trans.extend(mid_trans)
            if len(up_trans) > 0:
                cur_rep_trans.append('}')
            if len(down_trans) > 0:
                cur_rep_trans.append('}')
            set_map_rep_map[idx] = cur_rep_trans

    # replace
    out_words = []
    idx = 0
    while idx < len(words):
        if idx in set_map_rep_map:
            out_words.extend(set_map_rep_map[idx])
            idx = set_map[idx][3]
        else:
            out_words.append(words[idx])
        idx += 1
    return ' '.join(out_words)

def post_token_remove(transcription):
    words = transcription.split()
    global __post_remove_tokens_list
    out_words = []
    for word in words:
        if word in __post_remove_tokens_list:
            continue
        out_words.append(word)
    return ' '.join(out_words)

def reverse_transcription(transcription, matrix_left_right=True, name=""):
    """Input transcription format: \\frac { a } { c } + \\sqrt [ 3 ] { d }
    """
    transcription = post_line_replace(transcription)
    transcription = post_token_remove(transcription)
    transcription, bad_brace_trans = post_norm_brace(transcription)
    transcription = convert_chemical_equation(transcription)
    transcription, bad_matrix_trans = convert_matrix(transcription, matrix_left_right, name)
    transcription = post_word_replace(transcription)
    transcription = reverse_convert_chinese_to_uc(transcription)
    return transcription, bad_brace_trans or bad_matrix_trans


if __name__ == "__main__":
    """使用说明：
        1. 如果是模型输出的结果，可直接输入到reverse_transcription函数中后处理，该函数的作用是提供可渲染的合法LaTeX
        2. 如果是人工标注的结果，需要先输入到parse_transcription中进行规整，然后将规整后的结果输入到reverse_transcription函数中
    """
    # from transcription import parse_transcription
    # words = parse_transcription("\\begin{cases} dada a \\ b \\neq \\end{cases}")
    transcription, bad_trans = reverse_transcription('\\overset { \\underset { aa } { = } } \\rightarrow \\oversetfrown a \\pxsbx \\pxdy \\smear \\unk \\jump \\enter | | \\beginmatrix \ding 7 2 \ding 7 3 a + b \\endmatrix | |')
    print(transcription)
