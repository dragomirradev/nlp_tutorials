import json
import sys

input_fname = sys.argv[1]

with open(input_fname, 'r', encoding='iso-8859-1') as fp:
    for line in fp:
        label, *sentence = line.strip().split()
        main_label, sub_label = label.split(':')
        print(json.dumps({'main_category': main_label,
                          'sub_category': sub_label,
                          'question': ' '.join(sentence)}))


