# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:27:17 2020

@author: charl
"""

import argparse
import json
from Levenshtein import distance as levenshtein_distance

with open("config.json", 'r') as fp:
    CONFIG = json.load(fp)
    fp.close()

CONFIG = CONFIG['SIMPLIFIED_2']

def decode_p(p, INPUT):
    param_list = p['__concat__']
    res = ''
    
    for param in param_list:
        res += decode_e(param, INPUT)
        
    return res

def decode_e(e, INPUT):
    options = {
            '__sub_str__': decode_substr,
            '__to_case__': decode_to_case,
            '__get_token__': decode_get_token,
            '__swap__': decode_swap,
            '__const_str_c__': decode_const_str_c,
            '__const_str_w__': decode_const_str_w
        }
    key = list(e.keys())[0]
    return options[key](e[key], INPUT)

def decode_substr(k1_k2, INPUT):
    k1 = k1_k2[0]
    k2 = k1_k2[1]
    
    return INPUT[k1:k2]

def decode_to_case(s, INPUT):
    options = {
            CONFIG['s'][0]: decode_proper,
            CONFIG['s'][1]: decode_upper,
            CONFIG['s'][2]: decode_lower
        }
    
    return options[s](INPUT)

def decode_swap(i1_i2_t, INPUT):
    options = {
            CONFIG['t'][0]: decode_swap_number,
            CONFIG['t'][1]: decode_swap_digit,
            CONFIG['t'][2]: decode_swap_word,
            CONFIG['t'][3]: decode_swap_char
        }
    i1 = i1_i2_t[0]
    i2 = i1_i2_t[1]
    t = i1_i2_t[2]
    
    return options[t](i1, i2, INPUT)

def decode_lower(in_str):
    return in_str.lower()

def decode_upper(in_str):
    return in_str.upper()

def decode_proper(in_str):
    new_str = ''
    new_word = True
    for c in in_str:
        if c.isspace():
            new_word = True
            new_str += c
        elif new_word:
            new_str += c.upper()
            new_word = False
        else:
            new_str += c.lower()
        
    
    return new_str

def decode_get_token(i_t, INPUT):
    options = {
            CONFIG['t'][0]: decode_get_number,
            CONFIG['t'][1]: decode_get_digit,
            CONFIG['t'][2]: decode_get_word,
            CONFIG['t'][3]: decode_get_char
        }
    
    i = i_t[0]
    t = i_t[1]
    
    return options[t](i, INPUT)

def decode_const_str_c(c, INPUT):
    return ''.join(c)

def decode_const_str_w(w, INPUT):
    return ''.join(w)

def decode_swap_number(i1, i2, in_str):
    new_str = in_str.split()
    filtered_str = [[i, x] for i, x in enumerate(new_str) if x.isnumeric()]
    real_i1 = filtered_str[i1][0]
    real_i2 = filtered_str[i2][0]
    new_str[real_i1], new_str[real_i2] = new_str[real_i2], new_str[real_i1]
    return ' '.join(new_str)

def decode_swap_digit(i1, i2, in_str):
    filtered_str = [[i, x] for i, x in enumerate(in_str) if x.isnumeric()]
    real_i1 = filtered_str[i1][0]
    real_i2 = filtered_str[i2][0]
    in_str[real_i1], in_str[real_i2] = in_str[real_i2], in_str[real_i1]
    return ' '.join(in_str)

def decode_swap_word(i1, i2, in_str):
    new_str = in_str.split()
    new_str[i1], new_str[i2] = new_str[i2], new_str[i1]
    return ' '.join(new_str)

def decode_swap_char(i1, i2, in_str):
    new_str = list(in_str)
    new_str[i1], new_str[i2] = new_str[i2], new_str[i1]
    return ''.join(new_str)

def decode_get_number(i, in_str):
    new_str = in_str.split()
    filtered_str = [x for i, x in enumerate(new_str) if x.isnumeric()]
    return filtered_str[i]

def decode_get_digit(i, in_str):
    filtered_str = [x for ind, x in enumerate(in_str) if x.isnumeric()]
    return filtered_str[i]

def decode_get_word(i, in_str):
    new_str = in_str.split()
    return new_str[i]

def decode_get_char(i, in_str):
    return in_str[i]

def check_consistency(program, INPUT, OUTPUT):
    return levenshtein_distance(decode_p(program, INPUT), OUTPUT)


if __name__ == '__main__':
    ## Parsing arguments
    parser = argparse.ArgumentParser(description='Parser for a program described as a JSON')
    parser.add_argument('input_string')
    parser.add_argument('file')
    args = parser.parse_args()
    
    INPUT = args.input_string
    ## Loading data from input file
    with open(args.file, 'r') as f:
        ps = json.load(f)
    ## Decoding the program (or programs)
    for p in ps:
        res = decode_p(p, INPUT)
        print(res)