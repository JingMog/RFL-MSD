import os
import argparse

def main(args):
    vocab_path = args.output
    train_cs_string_path = args.train_input
    valid_cs_string_path = args.valid_input

    pre_define_words = ['\\unk', '<s>', '</s>', '\\enter', '\\jump', '\\space']
    vocab_list = ['>@']
    valid_cs_string = []
    with open(valid_cs_string_path, 'r') as f:
        valid_cs_string = f.readlines()
    for string in valid_cs_string:
        label = string.strip().split('\t')[1]
        label = label.split('—')[0]
        unit = label.split(' ')
        for u in unit:
            if u not in vocab_list and u not in pre_define_words:
                vocab_list.append(u)


    with open(train_cs_string_path, 'r') as f:
        valid_cs_string = f.readlines()
    for string in valid_cs_string:
        label = string.strip().split('\t')[1]
        label = label.split('—')[0]
        unit = label.split(' ')
        for u in unit:
            if u not in vocab_list and u not in pre_define_words:
                vocab_list.append(u)

    vocab_list = pre_define_words + sorted(vocab_list)
    assert len(vocab_list) == len(set(vocab_list))
    with open(vocab_path, 'w') as f:
        for i in range(len(vocab_list)):
            f.write(vocab_list[i] + '\t' + str(i) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("-train_input", type=str, default='./result/train_cs_string.txt')
    parser.add_argument("-valid_input", type=str, default='./result/valid_cs_string.txt')
    parser.add_argument("-output", type=str, default='./result/vocab.txt')
    args = parser.parse_args()
    main(args)



