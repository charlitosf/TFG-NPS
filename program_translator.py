# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:01:40 2020

@author: charl
"""
import argparse
import json
from tensorflow.keras.preprocessing.text import Tokenizer

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
    #print(res)
    return tokenizer.texts_to_sequences(res)
    

def RNN2JSON(rnn):
    print(rnn)
    programs = tokenizer.sequences_to_texts(rnn)
    programs = [p.split() for p in programs]
    res = []
    for p in programs:
        actual = {}
        if p[0] == CONFIG['methods']['concat']:
            actual[p[0]] = parse_concat_params(p[1:])
        res.append(actual)
    return res

def parse_concat_params(params):
    res = []
    actual = ''
    actual_method = ''
    for param in params:
        if param in CONFIG['methods'].values():
            
            if not actual == '':
                if len(actual[actual_method]) == 1:
                    actual[actual_method] = actual[actual_method][0]
                res.append(actual)
                
            actual_method = param
            actual = {
                    param: []
                }
        else:
            actual[actual_method].append(param)
    
    if len(actual[actual_method]) == 1:
        actual[actual_method] = actual[actual_method][0]
    res.append(actual)
    
    return res
    

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