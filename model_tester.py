# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:14:47 2021

@author: charl
"""
import experiment_runner as er
import program_translator as pt
import json
import numpy as np

with open("config.json", 'r') as fp:
    CONFIG = json.load(fp)
    fp.close()
    
CONFIG = CONFIG['DEFAULT']

def string_to_RNNinput(s):
    s = list(s)
    res = np.zeros((1, len(s)), dtype=np.int32)
    for idx_c, c in enumerate(s):
        res[ 0 ][ idx_c ] = char_to_int_intent[c]
    return res


if __name__ == "__main__":
    parser = er.getParser("Tester")
    args = parser.parse_args()
    
    model = er.get_model_from_args(args)
    
    tokenizer_program = pt.get_tokenizer()
    char_to_int_program = tokenizer_program.word_index
    
    tokenizer_io = er.getIOtokenizer()
    char_to_int_intent = tokenizer_io.word_index
    
    
    INPUT_PROGRAM = np.zeros((1, CONFIG['MAX_PROGRAM_SIZE']), dtype=np.int32)
    INPUT_PROGRAM[ 0 ][ 0 ] = char_to_int_program['<SOP>']
        
    while True:
        i_word = input("Enter input: ")
        o_word = input("Enter output: ")
        
        I_WORD = string_to_RNNinput(i_word)
        O_WORD = string_to_RNNinput(o_word)
        
        prediction = model([I_WORD, O_WORD, INPUT_PROGRAM])
        
        predictionChars = pt.rnn2list(prediction)
        
        print(predictionChars[0])