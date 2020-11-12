# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:09:52 2020

@author: charl
"""
import random
#random.seed(1235)
import string
import json
import argparse

MAX_E_CONCAT = 5
INPUT_STR_SIZE = 7
MAX_STR_SIZE = 15
MIN_STR_SIZE = 2

c = [string.ascii_uppercase, string.ascii_lowercase, string.digits, ' ', '.']
t = ["number", "digit", "word", "char"]
s = ["proper", "all_caps", "lower"]

def gen_p():
    amount_e = random.randint(1, MAX_E_CONCAT)
    res = {
            "concat" : []
        }
    for i in range(amount_e):
        res["concat"].append(gen_e())

    return res

def gen_e():
    choices = [gen_f, gen_n, gen_const_str_w, gen_const_str_c]
    choice = random.choice(choices)
    
    return choice()

def gen_f():
    res = {
        "sub_str": []
        }
    k1 = random.randint(0, INPUT_STR_SIZE-1)
    k2 = random.randint(k1, INPUT_STR_SIZE-1)
    res["sub_str"].append(k1)
    res["sub_str"].append(k2)
    
    return res

def gen_n():
    choices = [gen_get_token, gen_to_case, gen_swap]
    choice = random.choice(choices)
    
    return choice()
    
def gen_const_str_w():
    res = {}

    res["const_str_w"] = gen_word()
    
    return res

def gen_const_str_c():
    res = {}
    res["const_str_c"] = gen_character()
    
    return res

def gen_get_token():
    res = {
            "get_token": []
        }
    
    res_i = gen_index()
    res_type = gen_type()
    res["get_token"].append(res_i)
    res["get_token"].append(res_type)
    
    return res

def gen_to_case():
    res = {}
    res["to_case"] = gen_case()
    
    return res

def gen_swap():
    res = {
            "swap": []
        }
    
    res["swap"].append(gen_index())
    res["swap"].append(gen_index())
    res["swap"].append(gen_type())
    
    return res

def gen_type():
    return random.choice(t)

def gen_case():
    return random.choice(s)

def gen_index():
    return random.randint(-INPUT_STR_SIZE, INPUT_STR_SIZE-1)

def gen_word():
    length = range(random.randint(MIN_STR_SIZE, MAX_STR_SIZE))
    return ''.join([gen_character() for i in length])

def gen_character():
    return random.choice(random.choice(c))

if __name__ == "__main__":
    ## Argument parsing
    parser = argparse.ArgumentParser(description='Generator of random programs')
    parser.add_argument('-s', '--save', metavar='FILENAME')
    args = parser.parse_args()
    
    ## Generator call
    res = gen_p()
    
    ## Printing results
    print(json.dumps(res, sort_keys=True, indent=4))
    
    ## Storing results
    if (args.save):
        with open(args.save, 'w') as f:
            json.dump(res, f)
    