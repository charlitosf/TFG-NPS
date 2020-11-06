# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:27:17 2020

@author: charl
"""

import argparse
import json

def decode_p(p):
    param_list = p['concat']
    res = ''
    
    for param in param_list:
        res += decode_e(param)
        
    return res

def decode_e(e):
    options = {
            'sub_str': decode_substr,
            'to_case': decode_to_case,
            'get_token': decode_get_token,
            'swap': decode_swap,
            'const_str_c': decode_const_str_c,
            'const_str_w': decode_const_str_w
        }
    key = list(e.keys())[0]
    return options[key](e[key])

def decode_substr(k1_k2):
    k1 = k1_k2['k1']
    k2 = k1_k2['k2']
    
    return INPUT[k1:k2]

def decode_to_case(s):
    options = {
            'proper': decode_proper,
            'all_caps': decode_upper,
            'lower': decode_lower
        }
    
    return options[s](INPUT)

def decode_swap(i1_i2_t):
    options = {
            'number': decode_swap_number,
            'digit': decode_swap_digit,
            'word': decode_swap_word,
            'char': decode_swap_char
        }
    i1 = i1_i2_t['i1']
    i2 = i1_i2_t['i2']
    t = i1_i2_t['t']
    
    return options[t](i1, i2, INPUT)
    
    return ''

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

def decode_get_token(i_t):
    pass

def decode_const_str_c(c):
    return c

def decode_const_str_w(w):
    return w

def decode_swap_number(i1, i2, in_str):
    new_str = in_str.split()
    filtered_str = [[i, x] for i, x in enumerate(new_str) if x.isnumeric()]
    real_i1 = filtered_str[i1][0]
    real_i2 = filtered_str[i2][0]
    new_str[real_i1], new_str[real_i2] = new_str[real_i2], new_str[real_i1]
    return ' '.join(new_str)

def decode_swap_digit(i1, i2, in_str):
    filtered_str = [[i, x] for i, x in enumerate(in_str) if x.isNumeric()]
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

if __name__ == '__main__':
    ## Parsing arguments
    parser = argparse.ArgumentParser(description='Parser for a program described as a JSON')
    parser.add_argument('input_string')
    parser.add_argument('file')
    args = parser.parse_args()
    
    INPUT = args.input_string
    ## Loading data from input file
    with open(args.file, 'r') as f:
        p = json.load(f)
    
    ## Decoding the program
    res = decode_p(p)
    print(res)