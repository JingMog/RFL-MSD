# -*- coding: utf-8 -*-
import regex as re
import os
from collections import OrderedDict
import copy

def read_line_correct_map(line_correct_map):
    line_map = OrderedDict()
    if line_correct_map is not None:
        with open(line_correct_map, 'r') as slf:
            for line in slf:
                sep = '\r\n' if '\r\n' in line else '\n'
                segs = line.split(sep)[0].split('\t')
                if len(segs) < 2:
                    continue
                key = segs[0]
                val = set(segs[1:])
                if '' in val:
                    val.remove('')
                new_val = set()
                if key == '':
                    for v in val:
                        if v == 't e x t b f':
                            for tid in range(10):
                                tv = (v + '( %d )' % tid)
                                nv = tv + ' '
                                new_val.add(nv)
                        nv = v + ' '
                        new_val.add(nv)
                elif key == '<s>':
                    for v in val:
                        nv = v + ' '
                        new_val.add(nv)
                    key = key + ' '
                else:
                    for v in val:
                        nv = ' ' + v + ' '
                        new_val.add(nv)
                    key = ' ' + key + ' '                    
                if key in line_map:
                    line_map[key] = line_map[key] | new_val
                else:
                    line_map[key] = new_val
    return line_map

def read_word_correct_map(word_correct_map):
    word_map = OrderedDict()
    with open(word_correct_map, 'r') as wcmf:
        for line in wcmf:
            segs = line.strip().split('\t')
            if len(segs) < 1 or segs[0][0] == "#":
                continue
            if len(segs) == 1:
                rep_word = ""
            else:
                rep_word = ' '.join(segs[1:]).strip()
            if segs[0] in word_map:
                if word_map[segs[0]] != rep_word:
                    raise ValueError("%s dose exists, which value is %s, replace word is %s" % (segs[0], word_map[segs[0]], rep_word))
            word_map[segs[0]] = rep_word
    return word_map

def read_pre_word_correct_map(word_correct_map):
    with open(word_correct_map, 'r') as wcmf:
        word_map = wcmf.readlines()
        word_map = [w.strip() for w in word_map if w]
        return word_map

def get_rep_dict(rep_path=None):
    rep_dict = dict()
    if(rep_path is None):
        return rep_dict
    lines = open(rep_path).readlines()
    for line in lines:
        temp = line.split('\t')
        key = temp[0]
        value = temp[1].strip()
        rep_dict.update({key: value})

    return rep_dict

__post_line_correct_map = read_line_correct_map(
    os.path.join(os.path.dirname(__file__), 'post_line_correct.map'),
)

__post_word_correct_map = read_word_correct_map(
    os.path.join(os.path.dirname(__file__), 'post_word_correct.map')
)

__structure_word_correct_map = read_word_correct_map(
    os.path.join(os.path.dirname(__file__), 'structure_word_correct.map')
)

__pre_word_correct_map = read_pre_word_correct_map(
    os.path.join(os.path.dirname(__file__), 'pre_word_correct.map')
)

rep_dict = get_rep_dict(os.path.join(os.path.dirname(__file__), 'rep_dict.map'))

__splitout_regex_replace_map = OrderedDict((
    (r'[\s\\]begin[\s]*{[\s]*equation[\s]*[}]?', r' '),
    (r'[\s\\]en[asdfecxr]?[\s]*{[\s]*equation[\s]*[}]?', r' '),
    (r'[\s\\]begin[\s]*{[\s]*cases[\s]*[}]?', r' \{ \\beginmatrix '),
    (r'[\s\\]en[asdfecxr]?[\s]*{[\s]*cases[\s]*[}]?', r' \\endmatrix \}'),
    (r'[\s\\]because', r' \\because '),
    (r'[\s\\]therefore', r' \\therefore '),
    (r'[\s\\]frac', r' \\frac '),
    (r'[\s\\]angle', r' \\angle '),
    (r'[\s\\]quad', r' \\space '),
    (r'[\s\\]space', r' \\space '),
    (r'[\s\\]textcircle[d]?', r' \\textcircled '),
    (r'[\s\\]triangle', r' \\triangle '),
    (r'[\s\\]trianlge', r' \\triangle '),
    # (r'[\s\\]overset[\s]*[\\]?frown', r' \\oversetfrown '),
))

__splitout_replace_map = OrderedDict((
    (r'equation', r''),
    (r'< ERR >', r' \smear '),
    (r'/[/striangle/]', r' \triangle '),
    (r'<texttitleBegin>', r''),
    (r'<textBegin>', r''),
    (r'</textEnd>', r''),
    (r'\parallelogram', r'\pxsbx'),
    (r'\filler', r' \smear '),
    (r'\kongbai', r' \space '),
    (r"^^", r"\^"),
    (r"'", r"^{'}"),
    # 后处理显示的时候最好是添加上\left和\right的修饰符号
    (r"\begin{smallmatrix}", r"\beginmatrix "),
    (r"\begin{matrix}", r"\beginmatrix "),
    (r"\begin{Bmatrix}", r"\{ \beginmatrix "),
    (r"\begin{bmatrix}", r"[ \beginmatrix "),
    (r"\begin{pmatrix}", r"( \beginmatrix "),
    (r"\begin{Vmatrix}", r"| | \beginmatrix "),
    (r"\begin{vmatrix}", r"| \beginmatrix "),
    (r"\end{smallmatrix}", r"\endmatrix "),
    (r"\end{Bmatrix}", r"\endmatrix \} "),
    (r"\end{bmatrix}", r"\endmatrix ] "),
    (r"\end{pmatrix}", r"\endmatrix )"),
    (r"\end{Vmatrix}", r"\endmatrix | |"),
    (r"\end{vmatrix}", r"\endmatrix |"),
    (r"\numone", r"\textcircled{1}"),
    (r"\numtwo", r"\textcircled{2}"),
    (r"\numthree", r"\textcircled{3}"),
    (r"\numfour", r"\textcircled{4}"),
    (r"\numfive", r"\textcircled{5}"),
    (r"\numsix", r"\textcircled{6}"),
    (r"\numseven", r"\textcircled{7}"),
    (r"\numeight", r"\textcircled{8}"),
    (r"\numnine", r"\textcircled{9}"),
    (r"\orarraw", r"\overrightarrow { a }"),
    (r"\lVert", r" | | "),
    (r"\rVert", r" | | "),
    (r'//', r" \parallel "),
))

__splitout_remove_ops = OrderedDict((
    (r"\bancel", r"\bcancel"),
    (r"\\bancel", r"\bcancel"),
    (r"bancel", r"\bcancel"),
    (r"\bcanbel", r"\bcancel"),
    (r"\\bcanbel", r"\bcancel"),
    (r"bcanbel", r"\bcancel"),
))

__splitout_split_elements = OrderedDict((
    ("\\cm", "c m"),
    ("\\mm", "m m"),
    ("\\km", "k m"),
    ("\\lim", "l i m"),
    ("\\min", "m i n"),
    ("\\max", "m a x"),
    ("\\sin", "s i n"),
    ("\\cos", "c o s"),
    ("\\tan", "t a n"),
    ("\\sec", "s e c"),
    ("\\cot", "c o t"),
    ("\\csc", "c s c"),
    ("\\log", "l o g"),
    ("\\lg", "l g"),
    ("\\ln", "l n"),
    ("\\arcsin", "a r c s i n"),
    ("\\arccos", "a r c c o s"),
    ("\\arccot", "a r c c o t"),
    ("\\arctan", "a r c t a n"),
    ("\\sh", "s h"),
    ("\\det", "d e t"),
    ("\\downdownarrows", "\\downarrow \\downarrow"),
    ("\\upuparrows", "\\uparrow \\uparrow"),
    ("\\exp", "e x p"),
    ("\\ncong", "\\not \\cong"),
    ("\\ne", "\\not ="),
    ("\\neq", "\\not ="),
    ("\\nleq", "\\not \\leq"),
    ("\\nleqq", "\\not \\leqq"),
    ("\\ngeq", "\\not \\geq"),
    ("\\ngeqq", "\\not \\geqq"),

    ("\\nleqslant", "\\not \\leq"),
    ("\\nleqqslant", "\\not \\leqq"),
    ("\\ngeqslant", "\\not \\geq"),
    ("\\ngeqqslant", "\\not \\geqq"),
    ("\\nless", "\\not \\lt"),

    ("\\ngtr", "\\not \\gt"),
    ("\\nleftarrow", "\\not \\leftarrow"),
    ("\\nLeftarrow", "\\not \\Leftarrow"),
    ("\\nrightarrow", "\\not \\rightarrow"),
    ("\\nRightarrow", "\\not \\Rightarrow"),

    ("\\nsubset", "\\not \\subset"),
    ("\\nsubseteq", "\\not \\subseteq"),
    ("\\nsupset", "\\not \\supset"),
    ("\\nsupseteq", "\\not \\supseteq"),
    ("\\notin", "\\not \\in"),
    ("\\nexists", "\\not \\exists"),

    ("\\doteq", "\\dot { = }"),
    ("\\dotplus", "\\dot { + }"),
    ("\\ldots",      ". . ."),
    ("\\dots",      ". . ."),
    ("\\orarraw", "\\overrightarrow { a }"),
    ("\\nparallel", "\\not //"),
    ("\\nparalle", "\\not //"),
    ("\\lVert", " | | "),
    ("\\rVert", " | | "),
    ('\\|', ' | | '),
    ("\\minuscoloncolon", "- : :"),
    ("\\minuscolon",      "- :"),

    # boxed
    ("\\boxplus",    "\\boxed { + }"),
    ("\\boxminus",   "\\boxed { - }"),
    ("\\boxdot",     "\\boxed { \\cdot }"),
    ("\\boxtimes",   "\\boxed { \\times }"),

    ('\\bigotimes', '\\textcircled { \\times }'),
    ('\\bigoplus',  '\\textcircled { + }'),
    ('\\otimes',    '\\textcircled { \\times }'),
    ('\\oplus',     '\\textcircled { + }'),
    ('\\odot',      '\\textcircled { \\cdot }'),
    ('\\ominus',    '\\textcircled { - }'),
    ('\\odiv',      '\\textcircled { \\div }'),

))


__splitout_ignore_words = [
    '\\rm',
    '\\mbox',
    '\\mathrm',
    '\\Bigg',
    '\\small',
    '\\left',
    '\\right',
    '\\displaystyle',
    '\\scriptstyle',
    '\\scriptscriptstyle',
    '\\st',
    '\\bigg',
    '\\big',
    '\\mathord',
    '\\mathcal',
    '\\mathrel',
    '\\mathbin',
    '\\mathfrak',
    '\\mathtt',
    '\\mathclose',
    '\\mathbf',
    '\\mathopen',
    '\\mathbb',
    '\\mathsf',
    '\\mathbit',
    "\\math",
    "\\mathchar",
    "\\mathinner",
    '\\textstyle',
    '\\texttt',
    '\\textsf',
    '\\testvisiblespace',
    '\\textrm',
    '\\operatorname',
    '\\boldsymbol',
    '\\scriptstyle',
    '\\textstyle',
    '\\Big',
    '\\[',
    '\\]',
    '\\nolimits',
    # '\\limits', # TODO: for display, it should not be ignored
    '\\sqrthat',
    '\\text',
    '\\mathop',
    '\\!', # 缩小两个字符间的间距
    '\\bf',
    '$\\begin{equation}',
    '\\end{equation}$',
    '$',
    '\\begin',
    '\\end',
    '\\nulldelimiter'
]

__splitout_word_replace_map = OrderedDict((
    ("\\enter", "\\\\"),
    ('\\;', "\\space"),
    ('\\,', "\\space"),
    ('\\:', "\\space"),
    ('%', "\\%"),
    ('\\qquad', "\\space"),
    ('\\lt', '<'),
    ('\\gt', '>'),
    ('\\lbrack', '['),
    ('\\rbrack', ']'),
    ('\\colon', ':'),
    ('\\to', '\\rightarrow'),
    ('\\bar', '\\overline'),
    ('\\overrightline', '\\overline'),
    ('\\stackrel', '\\overset'),
    ('\\ne', '\\not ='),
    ('\\~', '\\sim'),
    ("'", '\\prime'),
    ('\\tfrac', '\\frac'),
    ('\\cfrac', '\\frac'),
    ('\\dfrac', '\\frac'),
    ('\\leqslant', '\\le'),
    ('\\geqslant', '\\ge'),
    ('\\leq', '\\le'),
    ('\\geq', '\\ge'),
    ('\\Sigma', '\\sum'),
    ('\\lvert', '|'),
    ('\\rvert', '|'),
    ('\\vert', '|'),
    ('\\mid', '|'),
    ('\\arrowvert', '|'),
    ('\\vartriangle', '\\triangle'),
    ('\\Delta', '\\triangle'),
    ('\\emptyset', '\\varnothing'),
    ('\\hat', '\\widehat'),
    ('\\tilde', '\\widetilde'),
    ('\\check', '\\widecheck'),
    ('\\lnot', '\\neg'),
    ('\\gvertneqq', '\\gneqq'),
    ('\\exist', '\\exists'),
    # TODO: 需要考虑是否保留
    ('\\Theta', '\\theta'),
    ('\\sharp', '#'),
    ('\\Alpha', 'A'),
    ('\\Beta', 'B'),
    ('\\Rho', 'P'),
    ('\\Mu', 'M'),
    ('\\Zeta', 'Z'),
    ('\\Nu', 'N'),
    ('\\Eta', 'H'),
    ('\\Epsilon', 'E'),
    ('\\Iota', 'I'),
    ('\\Chi', 'X'),
    ('\\Kappa', 'K'),
    ('\\kappa', 'k'),
    ('\\Tau', 'T'),
    ('\\^', '^'),
    ('\\"', '"'),
    ('\\perp', '\\bot'),
    ('\\prec', '<'),
    ('\\succ', '>'),
    ('\\succeq', '\\ge'),
    ('\\preceq', '\\le'),
    ('\\parallel', '//'),
    ('\\paralel', '//'),
    ('\\backsim', '\\sim'),
    ('\\Pi', '\\prod'),
    ('\\centerdot', '\\cdot'),
    ('\\vartheta', '\\phi'),
    ('\\lgroup',  '('),
    ('\\rgroup',  ')'),
    ('\\diagup', '/'),
    ('\\varsubsetneq', '\\subsetneq'),
    ('\\Lambda', '\\wedge'),
    ('\\land', '\\wedge'),
    ('\\ell', 'l'),
    ('\\dotsi', '\\cdots'),
    ('\\dotsm', '\\cdots'),
    ('\\ast', '*'),
    ('\\lor', '\\vee'),
    ("\\lmoustache", "\\int"), # TODO

    # big normliazation
    ('\\bigcap', '\\cap'),
    ('\\bigcup', '\\cup'),
    ('\\bigsqcap', '\\sqcap'),
    ('\\bigsqcup', '\\sqcup'),
    ('\\bigstar', '\\star'),
    ('\\bigvee', '\\vee'),
    ('\\bigwedge', '\\wedge'),
    ("\\intercal", "T"),

    # arrow normalization
    ('\\vec', '\\overrightarrow'),
    ('\\orarrow', '\\overrightarrow'),
    ('\\underleftarrow', '\\xleftarrow'),
    ('\\underrightarrow', '\\xrightarrow'),
    ('\\underleftrightarrow', '\\xleftrightarrow'),
    ('\\underLeftarrow', '\\xLeftarrow'),
    ('\\underRightarrow', '\\xRightarrow'),
    ('\\underLeftrightarrow', '\\xLeftrightarrow'),
    ('\\gets', '\\leftarrow'),
    ('\\iff', "\\Leftrightarrow"),
    ('\\implies', "\\Rightarrow"),
    ('\\longrightarrow', '\\rightarrow'),
    ('\\longleftarrow', '\\leftarrow'),
    ('\\longleftrightarrow', '\\leftrightarrow'),
    ('\\Longrightarrow', '\\Rightarrow'),
    ('\\Longleftarrow', '\\Leftarrow'),
    ('\\Longleftrightarrow', '\\Leftrightarrow'),
    ("\\xLongRightarrow", "\\xRightarrow"),
    ("\\xlongRightarrow", "\\xRightarrow"),
    ("\\xLongrightarrow", "\\xrightarrow"),
    ("\\xlongrightarrow", "\\xrightarrow"),
    ("\\xLongLeftarrow", "\\xLeftarrow"),
    ("\\xlongLeftarrow", "\\xLeftarrow"),
    ("\\xLongleftarrow", "\\xleftarrow"),
    ("\\xlongleftarrow", "\\xleftarrow"),
    ("\\xtofrom", "\\xrightleftarrows"),
    ("\\xrightequilibrium", "\\xrightleftharpoons"),
    ("\\xleftequilibrium", "\\xrightleftharpoons"),
    ("\\xtofrom", "\\xrightleftarrows"),

    # box normalization
    ("\\fbox", "\\boxed"),
    ("\\Box",  "\\square"),

    # other
    ('$\\quad $', '\\space'),
    ("\\smallint", '\\int'),

    # char normalization
    ('\\a', 'a'),
    ('\\b', 'b'),
    ('\\c', 'c'),
    ('\\d', 'd'),
    ('\\e', 'e'),
    ('\\f', 'f'),
    ('\\g', 'g'),
    ('\\h', 'h'),
    ('\\i', 'i'),
    ('\\j', 'j'),
    ('\\k', 'k'),
    ('\\l', 'l'),
    ('\\m', 'm'),
    ('\\n', 'n'),
    ('\\o', 'o'),
    ('\\p', 'p'),
    ('\\q', 'q'),
    ('\\r', 'r'),
    ('\\s', 's'),
    ('\\t', 't'),
    ('\\u', 'u'),
    ('\\v', 'v'),
    ('\\w', 'w'),
    ('\\x', 'x'),
    ('\\y', 'y'),
    ('\\z', 'z'),
    ('\\A', 'A'),
    ('\\B', 'B'),
    ('\\C', 'C'),
    ('\\D', 'D'),
    ('\\E', 'E'),
    ('\\F', 'F'),
    ('\\G', 'G'),
    ('\\H', 'H'),
    ('\\I', 'I'),
    ('\\J', 'J'),
    ('\\K', 'K'),
    ('\\L', 'L'),
    ('\\M', 'M'),
    ('\\N', 'N'),
    ('\\O', 'O'),
    ('\\P', 'P'),
    ('\\Q', 'Q'),
    ('\\R', 'R'),
    ('\\S', 'S'),
    ('\\T', 'T'),
    ('\\U', 'U'),
    ('\\V', 'V'),
    ('\\W', 'W'),
    ('\\X', 'X'),
    ('\\Y', 'Y'),
    ('\\Z', 'Z'),
    ('\\0', '0'),
    ('\\1', '1'),
    ('\\2', '2'),
    ('\\3', '3'),
    ('\\4', '4'),
    ('\\5', '5'),
    ('\\6', '6'),
    ('\\7', '7'),
    ('\\8', '8'),
    ('\\9', '9'),
    ('\\(', '('),
    ('\\)', ')'),
    ('\\=', '='),
    ('\\<', '<'),
    ('\\>', '>'),

))


def convert_transcription(transcription):
    transcription = ' ' + transcription + ' '
    transcription = transcription.replace('&lt;', '<')
    transcription = transcription.replace('&gt;', '>')
    transcription = transcription.replace('\\[', ' ')
    transcription = transcription.replace('\\]', ' ')
    transcription = transcription.replace('<ERR>', ' \\smear ')

    pattern1 = r"[<\s]INSERT[-_][A-Z]+[>\s]"
    transcription = re.sub(pattern1, r" ", transcription)

    pattern2 = "<ERR_BEG[A-Z]+>.*<ERR_END>"
    transcription = re.sub(pattern2, r" \\smear ", transcription)

    pattern3 = "<ERR_BEG[A-Z]+>"
    transcription = re.sub(pattern3, r" \\smear ", transcription)

    pattern4 = "<ERR_END>"
    transcription = re.sub(pattern4, r" \\smear ", transcription)

    pattern4 = "<ERR_END>"
    transcription = re.sub(pattern4, r" \\smear ", transcription)

    pattern5 = "<BIGERR>"
    transcription = re.sub(pattern5, r" \\smear ", transcription)

    transcription = transcription.replace('<BIGERR', ' \\smear ')
    transcription = transcription.replace('BIGERR>', ' \\smear ')
    transcription = transcription.replace('BIGERR', ' \\smear ')
    transcription = transcription.replace('<ERR', ' \\smear ')
    transcription = transcription.replace('ERR>', ' \\smear ')
    transcription = transcription.replace('ERR', ' \\smear ')
    transcription = transcription.replace('\\ifly_fil', ' \\smear ')
    transcription = transcription.replace('ifly_fil', ' \\smear ')


    # remove textit
    transcription = transcription.replace('\\textit', ' ') # TODO \\textit是手写体修饰符
    transcription = transcription.replace('\\it', ' ')

    pattern6 = '[^\\\\]textbf'
    transcription = re.sub(pattern6, r' \\textbf ', transcription)
    pattern7 = '^textbf'
    transcription = re.sub(pattern7, r' \\textbf ', transcription)

    pattern8 = r"\\[a-z]?kern[\s]*[{]?[+-]?[\d]+[mpe][utmx][}]?"
    transcription = re.sub(pattern8, r" \\space ", transcription)
    pattern9 = r"\\[hv]space[\s]*[{]?[+-]?[\d]+[mpe][utmx][}]?"
    transcription = re.sub(pattern9, r" \\space ", transcription)

    transcription = transcription.replace("\\kern", ' \\space ')
    transcription = transcription.replace("\\hspace", ' \\space ')

    pattern10 = r"\\para[\s]+llel"
    transcription = re.sub(pattern10, r" \\parallel ", transcription)

    pattern11 = r"\\right[\s]*\."
    transcription = re.sub(pattern11, r" ", transcription)

    pattern12 = r"\\left[\s]*\."
    transcription = re.sub(pattern12, r" ", transcription)

    pattern13 = r'\\ding[\s]*[{]*[\s]*7[\s]*2[\s]*[}]*'
    transcription = re.sub(pattern13, r' \\star', transcription)

    return transcription


def remove_start_block(info):
    info = re.sub(r"^[\s]+{", r"{", info) # 去除空格
    if len(info) <= 0:
        return info

    if info[0] != '{':
        return info
    
    left_brace_nums = 0
    right_brace_nums = 0
    for idx, char in enumerate(info):
        if char == '{':
            if idx == 0:
                left_brace_nums += 1
            elif idx > 0 and info[idx - 1] != '\\':
                left_brace_nums += 1
        elif char == '}':
            right_brace_nums += 1
        
        if left_brace_nums == right_brace_nums:
            return info[idx+1:]
    return info

def convert_matrix(transcription):
    # 处理array
    transcription = drop_brace_space(transcription)
    transcription = re.sub(r'[\s\\]begin{arra[y]?[}]?', r'\\begin{array}', transcription)
    transcription = re.sub(r'[\s\\]end{arra[y]?[}]?', r'\\end{array}', transcription)
    transcription = re.sub(r'[\s\\]beginarra[y]?', r'\\begin{array}', transcription)
    transcription = re.sub(r'[\s\\]endarra[y]?', r'\\end{array}', transcription)

    while r'\begin{array}' in transcription:
        begin_sidx = transcription.index(r'\begin{array}')
        begin_eidx = begin_sidx + len(r'\begin{array}')
        transcription = transcription[:begin_sidx] + r' \beginmatrix ' + remove_start_block(transcription[begin_eidx:])

    transcription = transcription.replace(r'\end{array}', r'\endmatrix ')

    # 处理cases
    transcription = transcription.replace(r'\begin{gathered}', r' \beginmatrix ')
    transcription = transcription.replace(r'\begin{cases}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begin{case}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begin{case s}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begin {cases}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'begin{cases}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begincases', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'begincases', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begincase ', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begi{cases}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begi{cases', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begin{equation} {cases}', r'\{ \beginmatrix ')
    transcription = transcription.replace(r'\begin{equation}{cases}', r'\{ \beginmatrix ')
    transcription = re.sub(r'\\begin[a-z]?ase[s]?[\s]*', r"\\{ \\beginmatrix ", transcription)
    transcription = re.sub(r'begin[a-z]?ase[s]?[\s]*', r"\\{ \\beginmatrix ", transcription)
    transcription = transcription.replace(r'\end{gathered}', r' \endmatrix ')
    transcription = transcription.replace(r'\\\end{cases}', r' \endmatrix ')
    transcription = re.sub('\\\\\\\\\s+\\\end{cases}', r' \\endmatrix ', transcription)
    transcription = transcription.replace(r'\end{cases}', r' \endmatrix ')
    transcription = transcription.replace(r'\\\end{cases', r' \endmatrix ')
    transcription = transcription.replace(r'\end{cases', r' \endmatrix ')
    transcription = transcription.replace(r'\\\nd{cases}', r' \endmatrix ')
    transcription = transcription.replace(r'\\\nd{cases', r' \endmatrix ')
    transcription = transcription.replace(r'\en{cases}', r' \endmatrix ')
    transcription = transcription.replace(r'\en{cases', r' \endmatrix ')
    transcription = transcription.replace(r'\e{cases}', r' \endmatrix ')
    transcription = transcription.replace(r'\e{cases', r' \endmatrix ')
    transcription = transcription.replace(r'end{cases}', r' \endmatrix ')
    transcription = re.sub(r'\\end[a-z]?ase[s]?[\s]*', r"\\endmatrix ", transcription)
    transcription = re.sub(r'end[a-z]?ase[s]?[\s]*', r"\\endmatrix ", transcription)
    transcription = transcription.replace(r'\hfill', r' ')

    transcription = re.sub(r'(\\begin\{(equation|matrix)\})*[\s]*\\begin\{split\}', r' \\beginaligned ', transcription)
    transcription = re.sub(r'\\end\{split\}[\s]*(\\end\{(equation|matrix)\})*', r' \\endaligned ', transcription)
    transcription = re.sub(r'(\\begin\{(equation|matrix)\})*[\s]*\\begin\{aligned\}', r' \\beginaligned ', transcription)
    transcription = re.sub(r'\\end\{aligned\}[\s]*(\\end\{(equation|matrix)\})*', r' \\endaligned ', transcription)

    transcription = transcription.replace(r'\begin{split}', r' \beginaligned ')
    transcription = transcription.replace(r'\end{split}', r' \endaligned ')


    # 处理matrix
    transcription = drop_brace_space(transcription)
    transcription = re.sub(r'[\s\\]begin{mat[a-z]*[}]?', r' \\beginmatrix ', transcription)
    transcription = re.sub(r'[\s\\]begin{Bmat[a-z]*[}]?', r' \\{ \\beginmatrix ', transcription)
    transcription = re.sub(r'[\s\\]begin{bmat[a-z]*[}]?', r' [ \\beginmatrix ', transcription)
    transcription = re.sub(r'[\s\\]begin{pmat[a-z]*[}]?', r' ( \\beginmatrix ', transcription)
    transcription = re.sub(r'[\s\\]begin{Vmat[a-z]*[}]?', r' | | \\beginmatrix ', transcription)
    transcription = re.sub(r'[\s\\]begin{vmat[a-z]*[}]?', r' | \\beginmatrix ', transcription)
    transcription = re.sub(r'[\s\\]end{mat[a-z]*[}]?', r' \\endmatrix ', transcription)
    transcription = re.sub(r'[\s\\]end{Bmat[a-z]*[}]?', r' \\endmatrix \\} ', transcription)
    transcription = re.sub(r'[\s\\]end{bmat[a-z]*[}]?', r' \\endmatrix ] ', transcription)
    transcription = re.sub(r'[\s\\]end{pmat[a-z]*[}]?', r' \\endmatrix )', transcription)
    transcription = re.sub(r'[\s\\]end{Vmat[a-z]*[}]?', r' \\endmatrix | | ', transcription)
    transcription = re.sub(r'[\s\\]end{vmat[a-z]*[}]?', r' \\endmatrix | ', transcription)

    def findall_substring_pos(string, sub):
        positions = list()
        start_pos = 0
        while True:
            pos = string.find(sub, start_pos, len(string))
            if pos >= 0:
                positions.append(pos)
                start_pos = pos + len(sub)
            else:
                break
        return positions

    def replace_skip(info, begin_trans=r'\beginmatrix', end_trans=r'\endmatrix'):
        begins_pos = findall_substring_pos(info, begin_trans)
        ends_pos = findall_substring_pos(info, end_trans)
        poses = [[pos, 1] for pos in begins_pos] + [[pos, -1] for pos in ends_pos]
        poses = sorted(poses, key=lambda x: x[0])

        replace_seps = list()
        start_pos, cum_val = None, 0
        for pos, val in poses:
            if (start_pos is None) and (val > 0):
                start_pos, cum_val = pos, val
            elif start_pos is not None:
                cum_val += val
                
                if cum_val == 0:
                    replace_seps.append([start_pos, pos])
                    start_pos, cum_val = None, 0
        
        result_str = ''
        head_pos = 0
        for sep_sp, sep_ep in replace_seps:
            result_str = result_str + info[head_pos:sep_sp]
            result_str = result_str + info[sep_sp:sep_ep].replace('\\\\', ' \\skip ')
            head_pos = sep_ep
        result_str = result_str + info[head_pos:]
        return result_str

    transcription = replace_skip(transcription, begin_trans=r'\beginmatrix', end_trans=r'\endmatrix')
    transcription = replace_skip(transcription, begin_trans=r'\beginaligned', end_trans=r'\endaligned')
    while re.search(r"\\beginmatrix[\s]+\\skip", transcription):
        transcription = re.sub(r"\\beginmatrix[\s]+\\skip", r" \\beginmatrix ", transcription)
    while re.search(r"\\skip[\s]+\\endmatrix", transcription):
        transcription = re.sub(r"\\skip[\s]+\\endmatrix", r" \\endmatrix ", transcription)

    while re.search(r"\\beginaligned[\s]+\\skip", transcription):
        transcription = re.sub(r"\\beginaligned[\s]+\\skip", r" \\beginaligned ", transcription)
    while re.search(r"\\skip[\s]+\\endaligned", transcription):
        transcription = re.sub(r"\\skip[\s]+\\endaligned", r" \\endaligned ", transcription)

    transcription = re.sub(r"\\beginmatrix[\s]+\\endmatrix", r" ", transcription)
    return transcription

def convert_chinese_to_uc(transcription):
    result = list()
    for char in transcription:
        if len(char.encode('utf-8')) > 1:
            body = char.encode('unicode_escape')[2:].decode('ascii').upper()
            if len(body) < 4:
                body = '0' * (4-len(body)) + body
            if len(body) == 4:
                char = '\\UC_' + body
            else:
                if len(body) < 8:
                    body = '0' * (8-len(body)) + body
                char = '\\LUC_' + body
        result.append(char)
    return ''.join(result)

def split_to_words(transcription):
    global __splitout_split_elements
    words = list()
    cur_word = ''
    for char in transcription:
        cur_word += char

        if cur_word.startswith('\\UC_'):
            if len(cur_word) == 8:
                words.append(cur_word)
                cur_word = ''
        elif cur_word.startswith('\\LUC_'):
            if len(cur_word) == 13:
                words.append(cur_word)
                cur_word = ''
        elif cur_word in ['\\\\', '\\{', '\\}', '\\$', '\\_', '\\;', '\\!', '\\|', '\\,', '\\:', '\\%', '\\&', '\\#']:
            words.append(cur_word)
            cur_word = ''
        elif (char.lower() in 'abcdefghijklmnopqrstuvwxyz'):
            if not cur_word.startswith('\\'):
                words.append(cur_word)
                cur_word = ''
        else:
            words.append(cur_word[:-1])
            if char == '\\':
                cur_word = char
            else:
                words.append(char)
                cur_word = ''
    words.append(cur_word)

    words = [word.strip() for word in words if word not in ['', ' ']]
    tmp_words = words
    words = list()
    for word in tmp_words:
        if word in __splitout_split_elements:
            words.extend(__splitout_split_elements[word].split(' '))
        else:
            words.append(word)

    return words


def replace_words(words, ignore_list=[]):
    global __splitout_ignore_words, __splitout_word_replace_map
    tmp_words = words
    words = list()
    for word in tmp_words:
        if word in __splitout_ignore_words + ignore_list:
            continue
        if word in __splitout_word_replace_map:
            word = __splitout_word_replace_map[word]
        words.append(word)
    return words

class Node:
    __subp_ops = [
        '_',
        '^',
        '\\overline',
        '\\underline',
        '\\overbrace',
        '\\overrightarrow',
        '\\overleftarrow',
        '\\Overrightarrow',
        '\\overleftrightarrow',
        '\\overgroup',
        '\\overlinesegment',
        '\\overleftharpoon',
        '\\overrightharpoon',
        '\\undergroup',
        '\\underlinesegment',
    ]
    __uni_ops = [
        '\\sqrt', # 根号
        '\\textbf', # 题号
        '\\textit', # 斜体
        '\\textcircled', # 带圈标号
        '\\dot', # 符号上有点
        '\\ddot',  # 符号上有两个点
        '\\widehat', #　符号上有尖括号
        '\\widetilde', #　符号上有波浪
        '\\utilde',
        "\\grave", # 符号上有顿号
        "\\boxed", # 方框
    ]
    __xsub_ops={
        "\\xleftarrow": "\\leftarrow",
        "\\xrightarrow": "\\rightarrow",
        "\\xLeftarrow":  "\\Leftarrow",
        "\\xRightarrow": "\\Rightarrow",
        "\\xleftrightarrow": "\\leftrightarrow",
        "\\xLeftrightarrow": "\\Leftrightarrow",
        "\\xhookleftarrow": "\\hookleftarrow",
        "\\xhookrightarrow": "\\hookrightarrow",
        "\\xmapsto": "\\mapsto",
        "\\xrightharpoondown": "\\rightharpoondown",
        "\\xrightharpoonup": "\\rightharpoonup",
        "\\xleftharpoondown": "\\leftharpoondown",
        "\\xleftharpoonup": "\\leftharpoonup",
        "\\xrightleftharpoons": "\\rightleftharpoons",
        "\\xleftrightharpoons": "\\leftrightharpoons",
        "\\xlongequal": "=",
        "\\xtwoheadrightarrow": "\\twoheadrightarrow",
        "\\xtwoheadleftarrow": "\\twoheadleftarrow",
        "\\xrightleftarrows": "\\rightleftarrows",
        "\\xleftrightarrows": "\\leftrightarrows",
    }
    __bin_ops=[
        "\\frac",
        "\\binom",
        "\\underset",
        "\\overset"
    ]
    def __init__(self, parent=None, need_brace=False):
        self.parent = parent
        self.need_brace = need_brace
        self.words = list()
    
    def append(self, word):
        self.words.append(word)

    def fix_op_nest(self):
        word_idx = len(self.words) - 1
        while word_idx >= 0:
            word = self.words[word_idx]
            if isinstance(word, Node):
                word.fix_op_nest()
            else:
                if (word in Node.__subp_ops) or \
                    (word in Node.__uni_ops) or \
                    (word in Node.__bin_ops) or \
                    (word in Node.__xsub_ops):
                    parent_node = Node(self, need_brace=False)
                    parent_node.words.append(self.words[word_idx])
                    if word == '\\sqrt':
                        left_words = fix_sqrt_nest(parent_node, self.words[word_idx+1:])
                    elif word in Node.__xsub_ops:
                        left_words = fix_xsub_nest(parent_node, self.words[word_idx+1:], Node.__xsub_ops[word])
                    elif (word in Node.__subp_ops) or (word in Node.__uni_ops):
                        left_words = fix_uni_sub_nest(parent_node, self.words[word_idx+1:])
                    elif word == '\\frac':
                        left_words = fix_frac_nest(parent_node, self.words[word_idx+1:])
                    elif word == '\\underset' or word == '\\overset':
                        left_words = fix_frac_nest(parent_node, self.words[word_idx+1:])
                        fix_over_under_set_nest(parent_node)
                    elif word == '\\binom':
                        left_words = fix_bin_nest(parent_node, self.words[word_idx+1:])
                        parent_node.words[1].need_brace = False
                        parent_node.words[2].need_brace = False
                        parent_node.words.insert(2, '\\\\')
                        parent_node.words = ['(', '\\beginmatrix'] + parent_node.words[1:] + ['\\endmatrix', ')']
                    else:
                        left_words = fix_bin_nest(parent_node, self.words[word_idx+1:])
                    self.words = self.words[:word_idx] + [parent_node] + left_words
            word_idx -= 1

    def flatten(self):
        result = list()
        for word in self.words:
            if isinstance(word, Node):
                result.extend(word.flatten())
            else:
                result.append(word)

        if self.need_brace:
            result.insert(0, '{')
            result.append('}')
        return result

    def remove_redundant_node(self):
        out_words = []
        for word in self.words:
            is_remove = False
            if isinstance(word, Node):
                word.remove_redundant_node()
                if not word.need_brace:
                    out_words.extend(word.words)
                    is_remove = True
            if not is_remove:
                out_words.append(word)
        self.words = out_words

    def norm_sup_sub_order(self):
        word_idx = 0
        while word_idx < len(self.words):
            if isinstance(self.words[word_idx], Node):
                self.words[word_idx].norm_sup_sub_order()
                if len(self.words[word_idx:]) >= 2 and isinstance(self.words[word_idx+1], Node):
                    self.words[word_idx+1].norm_sup_sub_order()
                    if len(self.words[word_idx].words) > 0 and self.words[word_idx].words[0] == "^" and len(self.words[word_idx+1].words) > 0 and self.words[word_idx+1].words[0] == "_":
                        if len(self.words[word_idx:]) >= 3:
                            self.words = self.words[:word_idx] + [self.words[word_idx+1], self.words[word_idx]] + self.words[word_idx+2:]
                        else:
                            self.words = self.words[:word_idx] + [self.words[word_idx+1], self.words[word_idx]]
                    word_idx += 1
            word_idx +=  1

    def norm_under_over_order(self):
        abnormal = False
        word_idx = 0
        while word_idx < len(self.words):
            if isinstance(self.words[word_idx], Node):
                self.words[word_idx].norm_under_over_order()
            elif self.words[word_idx] in ['\\overset', '\\underset']:
                self.words[word_idx+1].norm_under_over_order()
                self.words[word_idx+2].norm_under_over_order()
                up_node = None
                mid_node = None
                down_node = None
                if self.words[word_idx+1].words[0] in ['\\overset', '\\underset']: #只判断第一位是overset或者underset的情况
                    if self.words[word_idx] == '\\overset' and self.words[word_idx+1].words[0] == '\\overset':
                        up_node = self.words[word_idx+1].words[1]
                        mid_node = self.words[word_idx+1].words[2]
                        down_node = self.words[word_idx+2]
                    elif self.words[word_idx] == '\\overset' and self.words[word_idx+1].words[0] == '\\underset':
                        up_node = self.words[word_idx+1].words[2]
                        mid_node = self.words[word_idx+1].words[1]
                        down_node = self.words[word_idx+2]
                    elif self.words[word_idx] == '\\underset' and self.words[word_idx+1].words[0] == '\\overset':
                        up_node = self.words[word_idx+2]
                        mid_node = self.words[word_idx+1].words[1]
                        down_node = self.words[word_idx+1].words[2]
                    else: # self.words[word_idx] == '\\underset' and self.words[word_idx+1].words[0] == '\\underset':
                        up_node = self.words[word_idx+2]
                        mid_node = self.words[word_idx+1].words[2]
                        down_node = self.words[word_idx+1].words[1]
                elif self.words[word_idx+2].words[0] in ['\\overset', '\\underset']: #只判断第一位是overset或者underset的情况
                    if self.words[word_idx] == '\\overset' and self.words[word_idx+2].words[0] == '\\overset':
                        up_node = self.words[word_idx+1]
                        mid_node = self.words[word_idx+2].words[1]
                        down_node = self.words[word_idx+2].words[2]
                    elif self.words[word_idx] == '\\overset' and self.words[word_idx+2].words[0] == '\\underset':
                        up_node = self.words[word_idx+1]
                        mid_node = self.words[word_idx+2].words[2]
                        down_node = self.words[word_idx+2].words[1]
                    elif self.words[word_idx] == '\\underset' and self.words[word_idx+2].words[0] == '\\overset':
                        up_node = self.words[word_idx+2].words[1]
                        mid_node = self.words[word_idx+2].words[2]
                        down_node = self.words[word_idx+1]
                    else: # self.words[word_idx] == '\\underset' and self.words[word_idx+2].words[0] == '\\underset':
                        up_node = self.words[word_idx+2].word[2]
                        mid_node = self.words[word_idx+1].words[1]
                        down_node = self.words[word_idx+1]
                if ('\\overset' in  self.words[word_idx+1].words or '\\underset' in self.words[word_idx+1].words) \
                     and ('\\overset' in  self.words[word_idx+2].words or '\\underset' in self.words[word_idx+2].words):
                    abnormal = True
                if up_node and mid_node and down_node:
                    up_node = copy.deepcopy(up_node)
                    mid_node = copy.deepcopy(mid_node)
                    down_node = copy.deepcopy(down_node)
                    self.words[word_idx] = '\\underset'
                    self.words[word_idx+1].parent = self
                    self.words[word_idx+1].words = down_node.words
                    up_node.parent = self.words[word_idx+2]
                    mid_node.parent = self.words[word_idx+2]
                    self.words[word_idx+2].parent = self
                    self.words[word_idx+2].words = ['\\overset', up_node, mid_node]
                word_idx += 2
            word_idx += 1
        return abnormal

def fix_uni_sub_nest(parent, words):
    if len(words) <= 0:
        sub_node = Node(parent, need_brace=True)
        left_words = words
    else:
        sub_node = Node(parent, need_brace=True)
        sub_node.words.append(words[0])
        left_words = words[1:]
    parent.words.append(sub_node)
    return left_words


def fix_sqrt_nest(parent, words): # 处理sqrt中[]的问题
    if len(words) > 0 and words[0] == '[':
        if len(words) > 1 and words[1] != ']':
            para_end_idx = -1
            para_count = 1
            for idx, word in enumerate(words[1:]):
                if word == '[':
                    para_count += 1
                elif word == ']':
                    para_count -= 1
                if para_count == 0:
                    para_end_idx = idx + 1
                    break
            if para_end_idx > 0:
                parent.words.extend(words[:para_end_idx+1]) # parent 中包含了sqrt []这一段
                words = words[para_end_idx+1:]
        else:
            words = words[2:]
    return fix_uni_sub_nest(parent, words)

def fix_xsub_nest(parent, words, rep_word): # 处理xsub_ops中[]的问题, 用overset和underset替代
    parent.words[0] = '\\underset'
    underset_node1 = Node(parent, need_brace=True)
    underset_node2 = Node(parent, need_brace=True)
    underset_node2.words.append('\\overset')
    overset_node2 = Node(underset_node2, need_brace=True)
    overset_node2.words.append(rep_word) # mid word

    if len(words) > 0 and words[0] == '[':
        if len(words) > 1 and words[1] != ']':
            para_end_idx = -1
            para_count = 1
            for idx, word in enumerate(words[1:]):
                if word == '[':
                    para_count += 1
                elif word == ']':
                    para_count -= 1
                if para_count == 0:
                    para_end_idx = idx + 1
                    break
            if para_end_idx > 0:
                underset_node1.words.extend(words[1:para_end_idx])
                words = words[para_end_idx+1:]
        else:
            words = words[2:]
    left_words = fix_uni_sub_nest(underset_node2, words)
    underset_node2.words.append(overset_node2)
    fix_over_under_set_nest(underset_node2)
    underset_node2.need_brace = True
    parent.words.extend([underset_node1, underset_node2])
    # ignore blank
    fix_over_under_set_nest(parent)
    return left_words

def fix_bin_nest(parent, words):
    words = fix_uni_sub_nest(parent, words)
    words = fix_uni_sub_nest(parent, words)
    return words


def fix_frac_nest(parent, words):
    irregular_mode = False
    if len(words) >= 1:
        # 判断是否是\frac{12{4}}格式，
        if isinstance(words[0], Node):
            if len(words[0].words) >= 2:
                if (isinstance(words[0].words[0], Node) and \
                    all([not isinstance(word, Node) for word in words[0].words[1:]])) or \
                    (isinstance(words[0].words[-1], Node) and \
                    all([not isinstance(word, Node) for word in words[0].words[:-1]])):
                    if len(words)==1:
                        irregular_mode = True
                    elif not (isinstance(words[1], Node)):
                        irregular_mode = True
    if irregular_mode:
        first_item = words[0]
        left_words = words[1:]
        if isinstance(first_item.words[0], Node):
            sub_node1 = first_item.words[0]
            sub_node1.parent=parent
            sub_node1.need_brace=True
            
            sub_node2 = Node(parent, need_brace=True)
            sub_node2.words = first_item.words[1:]
        else:
            sub_node1 = Node(parent, need_brace=True)
            sub_node1.words = first_item.words[:-1]

            sub_node2 = first_item.words[-1]
            sub_node2.parent=parent
            sub_node2.need_brace=True
        parent.words.append(sub_node1)
        parent.words.append(sub_node2)
        return left_words
    else:
        return fix_bin_nest(parent, words)

def fix_over_under_set_nest(parent):
    parent.remove_redundant_node()
    if len(parent.words[1].words) == 0:
        parent.need_brace = False
        parent.words = parent.words[2].words
    elif len(parent.words[2].words) == 0:
        parent.need_brace = False
        parent.words = parent.words[1].words

def parse_tree(words):
    tree = Node()
    cur_node = tree
    for word in words:
        if word == '{':
            new_node = Node(cur_node)
            cur_node.words.append(new_node)
            cur_node = new_node
        elif word == '}':
            if cur_node.parent is not None:
                cur_node = cur_node.parent
        else:
            cur_node.words.append(word)
    return tree


def splitout_transcritpion(transcription,ignore_list=[]):
    global __splitout_replace_map, __splitout_remove_ops, __splitout_regex_replace_map
    for key, val in __splitout_regex_replace_map.items():
        transcription = re.sub(key, val, transcription)

    for key, val in __splitout_replace_map.items():
        transcription = transcription.replace(key, val)

    for key, val in __splitout_remove_ops.items():
        transcription = transcription.replace(key, val)
    transcription = re.sub(r"\\[a-z]?cancel", r"\\bcancel", transcription) # 替换所有cancel的内容，这一块印刷体不需要处理

    for remove_op in set(__splitout_remove_ops.values()):
        while remove_op in transcription:
            remove_pos = transcription.index(remove_op)
            transcription = transcription[:remove_pos] + ' \\smear ' + remove_start_block(transcription[remove_pos+len(remove_op):])
    
    words = split_to_words(transcription)
    words = replace_words(words, ignore_list=ignore_list)
    words_tree = parse_tree(words)
    words_tree.fix_op_nest()
    words_tree.norm_sup_sub_order()
    words_tree.remove_redundant_node()
    abnormal = words_tree.norm_under_over_order()
    words = words_tree.flatten()
    words = [word if word != '\\skip' else '\\\\' for word in words]
    return words, abnormal

def drop_brace_space(transcription):
    transcription = re.sub(r'\s+{', r'{', transcription)
    transcription = re.sub(r'{\s+', r'{', transcription)
    transcription = re.sub(r'}\s+', r'}', transcription)
    transcription = re.sub(r'\s+}', r'}', transcription)
    return transcription

def post_correct_mistakes(words):
    """
    correct some spelling mistakes in label which was generated by complex_process_ytzhang2.pl
    """
    global __post_line_correct_map, __post_word_correct_map
    # join words
    transcription = " ".join(words)
    # replace for contents
   
    for key, val in __post_line_correct_map.items():
        for v in val:
            while v in transcription:
                transcription = transcription.replace(v, key)

    # correct
    words  = transcription.split(" ")
    out_words = []
    for lab in words:
        if lab in __post_word_correct_map:
            new_lab = __post_word_correct_map[lab].split(' ')
        else:
            new_lab = [lab]
        new_lab = [item for item in new_lab if item not in ['', '\\']]
        out_words.extend(new_lab)
    return out_words

def pre_correct_mistakes(transcription):
    transcription = ' ' + transcription
    global __structure_word_correct_map, __pre_word_correct_map
    for word in __structure_word_correct_map:
        pattern = "\\" + word + r"(?=[^a-zA-Z])"
        transcription = re.sub(pattern, __structure_word_correct_map[word].replace("\\", "\\\\"), transcription)
    for symbol in __pre_word_correct_map:
        pattern = r"[\s]" + symbol[1:] + r"(?=[^a-zA-Z])"
        transcription = re.sub(pattern, symbol.replace("\\", "\\\\"), transcription)
    return transcription

def post_rep_unicode(transcription=None):
    global rep_dict
    for key, value in rep_dict.items():
        transcription = transcription.replace(key, ' ' + value + ' ')
    return transcription

def parse_transcription(transcription, ignore_list=['\\textbf'], simple_trans=False):
    # \textbf 表示题号，填空题中不需要题号，所以需要在ignore_list中添加\\textbf，以便规整过程中去掉该符号
    if transcription in ['ERR', '<ERR>', 'BIGERR', '<BIGERR>']:
        return ['\\smear']
    src_transcription = transcription
    if not simple_trans:
        transcription = pre_correct_mistakes(transcription)
    transcription = drop_color_codes(transcription)
    transcription = convert_transcription(transcription)
    transcription = convert_chinese_to_uc(transcription)
    transcription = post_rep_unicode(transcription)
    transcription = convert_matrix(transcription)
    words, abnormal = splitout_transcritpion(transcription, ignore_list)
    if abnormal:
        print('Source Transcription: %s\nAbnormal Transcription: %s' % (src_transcription, words))
    words = post_correct_mistakes(words)
    return words
    
def drop_color_codes(transcription):
    """ Attention: it is a very simple version to drop color codes, may not be definitely right
    """
    pattern1 = r'\\color[\s]*{[a-z,A-Z]*}'
    transcription = re.sub(pattern1, r' ', transcription)

    pattern2 = '\\color '
    transcription = transcription.replace(pattern2, ' ')
    return transcription

if __name__ == "__main__":
    # parse_transcription(" \\hspace {3pt} dad frac {ddaa}{\\BCANCEL{\\FRAC{3}{2}}} \\begin{array} \\begin{cases} \\frac{ad} {da} a\\\\bc\\\\d \\end{cases} \\end{array}\\BCANCEL{\\frac{1}{2}\\sqrt[2]{\\frac{2}{3}}}\\begin{matrix} c \\\\ e a \\enter b \\end{matix} a% b")
    # parse_transcription("\\begincass Ⅻ dada \\d c \\endcases> dada ac \\c INSERT_END \\ 测试结果、、| \\left \{ \\begin{matrix} a \\ b \\end{matrix} \\right.dada \\ddot { a } \\xlongequal{a} \\stackrel a =")
    # out_words = parse_transcription(ignore_list = [], transcription=" 、 \\ifly_fil\\textbf ①³ %   \\begin{matrix} \\begin{aligned} 50 \\\\ \\times 3\\\\ \\hline 150\\\\ \\end{aligned} \\end{matrix}<s>  \\{ \\beginmatrix  \\endmatrix \\{ \\beginmatrix x = - 1 \\\\ \\\\ \\endmatrix </s> \\{ \\beginmatrix x + y = 6 \\\\ \\endmatrix  \\begin{split} \n 20.0\\\\ - 1.8\ \ \n \hline 18.2\\ \n \end{split}  ∫ ∴ \\begincass Ⅻ dada \\d c \\endcases> dada ac \\c INSERT_END \\ 测试结果、、| \\left \{ \\begin{matrix} a \\ b \\end{matrix} \\right.dada \\ddot { a } \\xlongequal[\\sqrt[a_2]{3^2}] cd \\stackrel a = \\frac{}{2} \\overset{\\underset {d}{=}}{ c}")
    # out_words = parse_transcription(ignore_list = [], transcription="\\ding 7 2  \\underline{C_{2}H_{5}OH+CH_{3}COOH \\underset{\\Delta }{\\overset{H_{2}30_{4}l浓}CH_{2}CH_{3}LOOH+H_{2}O}}\\xlongleftarrow")
    # print(out_words)
    # out_words = parse_transcription(ignore_list = [], transcription="\\begin{matrix}\\begin{aligned}2 |\\underline{20\\space30} \\ 5 |\\underline{10\\space15} \\ 2\\space 3 \\ \\end{aligned} \\end{matrix}")
    # print(out_words)
    out_words = parse_transcription(ignore_list = [], transcription="我𬬻1")
    print(out_words)
    