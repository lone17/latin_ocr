import os
import re
import json
import random
import itertools
import cv2 as cv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from alphabet import *

def reprocess(s):
    s = re.sub(r'(\s|-|\.)\1+', r'\1', s)
    s = s.replace('->', '→')
    s = s.translate(str.maketrans(dict(zip('[{}];≒', '(()):='))))
    pt = '[^' + ''.join(alphabet) + ']'
    s = re.sub(pt, unknown, s)

    return s

def normalize_label(s, convert_table=convert_table):
    result = []
    for c in s:
        result.append(convert_table[c])
    return ''.join(result)

def normalize_labels_in_json(json_file, out):
    with open(json_file, 'r', encoding='utf8') as f:
        content = json.load(f)
    files = list(content.keys())
    labels = dict.fromkeys(files)
    for file in files:
        labels[file] = normalize_label(content[file])
    with open(out, 'w', encoding='utf8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)


def vectorize_string(string, index_table=alphabet_index):
    result = [index_table[letter] for letter in string]
    return np.array(result)

def encode_y(y, alphabet=alphabet):
    return [vectorize_string(s) for s in y]

def make_y_from_transcripts(transcripts, out):
    with open(transcripts, 'r', encoding='utf-8') as f:
        y = f.read().split('\n')
    y = [reprocess(s) for s in y]
    y = encode_y(y)
    y = pad_sequences(y, padding='post', value=blank_index)
    np.save(out, y)

def ctc_decoder(pred):
    out_best = list(np.argmax(pred[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    text = ''
    for i in out_best:
        if i < len(alphabet):
            text += alphabet[i]
    return text

# def foo(c):
#     error = [k for k, v in content.items() if c in v and k in files]
#     print(len(error))
#     random.shuffle(error)
#     if len(error) > 5:
#         print(error[:5])
#     return error
#
# def bar(err):
#     for e in err:
#         print(e)
#         os.startfile(os.path.normpath(data_dir + e))

# make_y_from_transcripts('transcripts_real.txt', 'y_real_2')
