# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:01:40 2020

@author: charl
"""
import argparse
import json
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np

with open("config.json", 'r') as fp:
        CONFIG = json.load(fp)
        fp.close()
CONFIG = CONFIG['DEFAULT']

tokenizer = Tokenizer(filters='',
                      lower=False,
                      #oov_token="<UNK>"
                      )

def train_tokenizer(tokenizer):
    tokenizer.fit_on_texts(["<EOP>", "<SOP>"])
    tokenizer.fit_on_texts(CONFIG['methods'].values())
    tokenizer.fit_on_texts(CONFIG['t'])
    tokenizer.fit_on_texts(CONFIG['s'])
    positives = [str(i) for i in range(CONFIG['MAX_STR_SIZE'])]
    negatives = [str(-i - 1) for i in range(CONFIG['MAX_STR_SIZE'])]
    tokenizer.fit_on_texts(positives)
    tokenizer.fit_on_texts(negatives)
    characters = [list(c) for c in CONFIG['c']]
    tokenizer.fit_on_texts(characters)
    return tokenizer

tokenizer = train_tokenizer(tokenizer)

def get_tokenizer():
    return tokenizer

def JSON2RNN_recurs(json):
    res = []
    if isinstance(json, dict):
        for key in json.keys():
            res.append(str(key))
            res += JSON2RNN_recurs(json[key])
    elif isinstance(json, list):
        for el in json:
            res += JSON2RNN_recurs(el)
    else:
        res.append(str(json))
    return res

def JSON2RNN(json):
    res = []
    for j in json:
        res.append(JSON2RNN_recurs(j))
    return tokenizer.texts_to_sequences(res)
    
def argmax(values):
    return np.argmax(values.numpy())

def cut_by_eop(s, leave_first_eop = False):
    for idx, n in enumerate(s):
        if tokenizer.index_word[n] == '<EOP>':
            return s[:idx + 1 if leave_first_eop else idx]
    return s

def toChar(l):
    return [tokenizer.index_word[c] for c in l]

def compute_rnn_program(rnn, to_char = False):
    prediction = tf.map_fn(fn=argmax, elems=rnn).numpy().astype('int32')
    prediction = cut_by_eop(prediction, to_char)
    if to_char:
        return toChar(prediction)
    return prediction

def RNN2JSON(rnn_programs):
    rnn_programs = tf.identity(rnn_programs)
    integer_programs = np.array(list(map(compute_rnn_program, rnn_programs))).astype('int32')
    programs = tokenizer.sequences_to_texts(integer_programs)
    programs = [p.split() for p in programs]
    res = []
    for p in programs:
        actual = {}
        if p[0] == CONFIG['methods']['concat']:
            actual[p[0]] = parse_concat_params(p[1:])
        else:
            raise Exception(p[0])
        res.append(actual)
    return res

def rnn2list(rnn):
    res = []
    for elem in rnn:
        predictionChars = compute_rnn_program(elem, True)
        res.append(predictionChars)
    return res

def parse_concat_params(params):
    res = []
    actual = ''
    actual_method = ''
    for param in params:
        if param in CONFIG['methods'].values():
            if not actual == '':
                res.append(check_method(actual, actual_method))
                
            actual_method = param
            actual = {
                    param: []
                }
        else:
            actual[actual_method].append(param)

    res.append(check_method(actual, actual_method))
    
    return res

def check_method(actual, actual_method):
    if len(actual[actual_method]) == 1:
        if actual_method in ['__const_str_c__', '__to_case__']:
            actual[actual_method] = actual[actual_method][0]
            return actual
        raise Exception(actual_method, actual[actual_method])
    if  (actual_method in ['__sub_str__', '__get_token__'] and len(actual[actual_method]) != 2) or \
        (actual_method == '__swap__' and len(actual[actual_method]) == 3) or \
        (actual_method == '__get_token__' and actual[actual_method][1] not in CONFIG['t']) or \
        (actual_method == '__swap__' and actual[actual_method][2] not in CONFIG['t']) or \
        (actual_method == '__to_case__' and actual[actual_method] not in CONFIG['s']) \
    :
        raise Exception(actual_method, actual[actual_method])
    
    return actual

def is_rnn_program_correct(program):
    try:
        RNN2JSON([program])
        return True
    except Exception:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program translator JSON - Number Array")
    parser.add_argument('-s', '--save', metavar='FILE', help='Path where the output will be stored')
    parser.add_argument('file', help='Path to input file to be translated')
    args = parser.parse_args()
    
    
    
    with open(args.file, 'r') as f:
        inpt = json.load(f)
        f.close()
        
    JSONtoRNN = isinstance(inpt[0], dict)
    if JSONtoRNN:
        res = JSON2RNN(inpt)
    else:
        res = RNN2JSON(inpt)
        
    
    if args.save is not None:
        with open(args.save, 'w') as f:
            json.dump(res, f)
            f.close()
    
    print(res)