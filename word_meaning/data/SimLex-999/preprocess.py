import json
import sys

input_fname = sys.argv[1]

line_vals = ['word1', 'word2', 'POS', 'SimLex999',
             'conc_w1', 'conc_w2', 'concQ',
             'Assoc_USF', 'SimAssoc333', 'SD_SimLex']

with open(input_fname, 'r', encoding='iso-8859-1') as fp:
    for line in fp:
        dict_ = dict(zip(line_vals, line.strip().split('\t')))
        print(json.dumps(dict_))

