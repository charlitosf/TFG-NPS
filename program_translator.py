# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:01:40 2020

@author: charl
"""
import argparse
import json
from tensorflow.keras.preprocessing.text import Tokenizer


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
    return res

def RNN2JSON(rnn):
    return str(rnn)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program translator JSON - Number Array")
    parser.add_argument('-s', '--save', metavar='FILE', help='Path where the output will be stored')
    parser.add_argument('file', help='Path to input file to be translated')
    parser.add_argument('JSONtoRNN', type=str2bool, help='True if input file is JSON, False if input file is number array (aka RNN)')
    args = parser.parse_args()
    
    with open(args.file, 'r') as f:
        if args.JSONtoRNN:
            inpt = json.load(f)
        else:
            inpt = f.readline()
        f.close()
        
    if not args.JSONtoRNN:
        inpt = inpt.split()
        inpt = [int(i) for i in inpt]
        
    if args.JSONtoRNN:
        res = JSON2RNN(inpt)
        print(res)
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(res)
        res = tokenizer.texts_to_sequences(res)
    else:
        res = RNN2JSON(inpt)
    
    
    if args.save is not None:
        with open(args.save, 'w') as f:
            f.write(res)
            f.close()
    
    print(res)