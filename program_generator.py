# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:09:52 2020

@author: charl
"""
import random
#random.seed(1235)
import json
import argparse

with open("config.json", 'r') as fp:
    CONFIG = json.load(fp)
    fp.close()
    
CONFIG = CONFIG['SIMPLIFIED_1']

MAX_E_CONCAT = CONFIG['MAX_E_CONCAT']
MAX_STR_SIZE = CONFIG['MAX_STR_SIZE']
MIN_STR_SIZE = CONFIG['MIN_STR_SIZE']

METHODS = CONFIG['methods']

c = CONFIG['c']
t = CONFIG['t']
s = CONFIG['s']

def get_intent_vocabulary():
    tam_intent_vocabulary = 0
    for s in CONFIG['c']:
        for c in s:
            tam_intent_vocabulary += 1
    return tam_intent_vocabulary

def gen_p():
    amount_e = random.randint(1, MAX_E_CONCAT)
    res = {
            METHODS['concat'] : []
        }
    for i in range(amount_e):
        res[METHODS['concat']].append(gen_e())

    return res

def gen_e():
    choices = [gen_f, gen_n, gen_const_str_w]
    
    choice = random.choice(choices)
    
    return choice()

def gen_f():
    res = {
            METHODS['sub_str']: []
        }
    k1 = random.randint(0, MAX_STR_SIZE-1)
    k2 = random.randint(k1, MAX_STR_SIZE-1)
    res[METHODS['sub_str']].append(k1)
    res[METHODS['sub_str']].append(k2)
    
    return res

def gen_n():
    choices = [gen_get_token, gen_to_case, gen_swap]
    choice = random.choice(choices)
    
    return choice()
    
def gen_const_str_w():
    res = {}

    res[METHODS['const_str_w']] = gen_word()
    
    return res

def gen_const_str_c():
    res = {}
    res[METHODS['const_str_c']] = [gen_character()]
    
    return res

def gen_get_token():
    res = {
            METHODS['get_token']: []
        }
    
    res_i = gen_index()
    res_type = gen_type()
    res[METHODS['get_token']].append(res_i)
    res[METHODS['get_token']].append(res_type)
    
    return res

def gen_to_case():
    res = {}
    res[METHODS['to_case']] = gen_case()
    
    return res

def gen_swap():
    res = {
            METHODS['swap']: []
        }
    
    res[METHODS['swap']].append(gen_index())
    res[METHODS['swap']].append(gen_index())
    res[METHODS['swap']].append(gen_type())
    
    return res

def gen_type():
    return random.choice(t)

def gen_case():
    return random.choice(s)

def gen_index():
    return random.randint(-MAX_STR_SIZE, MAX_STR_SIZE-1)

def gen_word():
    length = range(random.randint(MIN_STR_SIZE, MAX_STR_SIZE))
    return [gen_character() for i in length]

def gen_character():
    return random.choice(random.choice(c))

if __name__ == "__main__":
    ## Argument parsing
    parser = argparse.ArgumentParser(description='Generator of random programs')
    parser.add_argument('-s', '--save', metavar='FILENAME', help='Filename of file where output is to be saved')
    parser.add_argument('-i', '--iterations', metavar='ITERATIONS', help='Amount of generated programs', type=int, default=1)
    args = parser.parse_args()
    
    res = []
    ## Generator call
    for i in range(args.iterations):
        res.append(gen_p())
    
    ## Printing results
    print(json.dumps(res, sort_keys=True, indent=4))
    
    ## Storing results
    if (args.save):
        with open(args.save, 'w') as f:
            json.dump(res, f)