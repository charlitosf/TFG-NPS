# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:14:47 2021

@author: charl
"""
import experiment_runner as er
import program_translator as pt
import json
import numpy as np
import tensorflow as tf

with open("config.json", 'r') as fp:
    CONFIG = json.load(fp)
    fp.close()
    
CONFIG = CONFIG['DEFAULT']

def string_to_RNNinput(s, result):
    s = list(s)
    for idx_c, c in enumerate(s):
        result[ 0 ][ idx_c ] = char_to_int_intent[c]
    return result

def find_largest_index(l):
    max_value = 0.0
    best_i = -1
    
    for idx, n in enumerate(l):
        if n > max_value:
            max_value = n
            best_i = idx
    return best_i

def cut_by_eop(s):
    for idx, n in enumerate(s):
        if int_to_char_program[n] == '<EOP>':
            return s[:idx]
    return s

def toChar(l):
    return [int_to_char_program[c] for c in l]

if __name__ == "__main__":
    parser = er.getParser("Tester")
    args = parser.parse_args()
    
    model = er.getModel(train=args.train, examples_per_epoch=2 ** 15, batch_size=args.batch_size, epochs=args.epochs, evolution_graph=args.evolution_graph, saving_frequency=args.saving_frequency)
    
    tokenizer_program = pt.get_tokenizer()
    char_to_int_program = tokenizer_program.word_index
    int_to_char_program = tokenizer_program.index_word
    
    tokenizer_io = er.getIOtokenizer()
    char_to_int_intent = tokenizer_io.word_index
    
    
    INPUT_PROGRAM = np.zeros((1, CONFIG['MAX_PROGRAM_SIZE']), dtype=np.int32)
    INPUT_PROGRAM[ 0 ][ 0 ] = char_to_int_program['<SOP>']
        
    while True:
        i_word = input("Enter input: ")
        o_word = input("Enter output: ")
        
        I_WORD = np.zeros((1, len(i_word)), dtype=np.int32)
        O_WORD = np.zeros((1, len(o_word)), dtype=np.int32)
        
        I_WORD = string_to_RNNinput(i_word, I_WORD)
        O_WORD = string_to_RNNinput(o_word, O_WORD)
        
        prediction = model([I_WORD, O_WORD, INPUT_PROGRAM])

        prediction = tf.map_fn(fn=lambda values: np.argmax(values.numpy()), elems=prediction[0]).numpy().astype('int32')
        
        prediction = cut_by_eop(prediction)
        
        predictionChars = toChar(prediction)
        
        print(predictionChars)
        
        prediction = pt.RNN2JSON([prediction])[0]
        
        print(prediction)