# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:28:44 2020

@author: charl
"""
import neural_networks as nn
import program_generator as pg
import program_runner as pr
import program_translator as pt
import numpy as np
import tensorflow as tf
import json

from tensorflow.keras.preprocessing.text import Tokenizer

with open("config.json", 'r') as fp:
    CONFIG = json.load(fp)
    fp.close()

CONFIG = CONFIG['DEFAULT']


def getIOtokenizer():
    tokenizer = Tokenizer(filters='',
                      lower=False,
                      #oov_token="<UNK>"
                      )
    tokenizer.fit_on_texts(["<EOP>", "<SOP>"])
    characters = [list(c) for c in CONFIG['c']]
    tokenizer.fit_on_texts(characters)
    return tokenizer

def generator(tam_lote = 32):
    while True:
        i_words = []
        o_words = []
        
        o_programs = []
        
        for _ in range(tam_lote):
            i_word = pg.gen_word()
            fails = True
            while fails:
                try:
                    o_program = pg.gen_p()
                    o_word = pr.decode_p(o_program, ''.join(input))
                    
                    fails = False
                except:
                    print("Execution failure, generating a new program")
            
            i_words.append(i_word)
            o_words.append(list(o_word))
            translated_program = pt.JSON2RNN([o_program])[0]
            o_programs.append(translated_program)
            
            
        max_longitud_iwords = max([len(word) for word in i_words])
        max_longitud_owords = max([len(word) for word in o_words])
        
        max_longitud_oprograms = max([len(word) for word in o_programs])
        
        
        
        I_WORDS = np.zeros((tam_lote, max_longitud_iwords), dtype=np.int32)
        O_WORDS = np.zeros((tam_lote, max_longitud_owords), dtype=np.int32)
        I_PROGRAMS = np.zeros((tam_lote, max_longitud_oprograms), dtype=np.int32)
        
        O_PROGRAMS = np.zeros((tam_lote, max_longitud_oprograms), dtype=np.int32)
        
        for idx_iw, w in enumerate(i_words):
            for idx_ic, c in enumerate(w):
                I_WORDS[ idx_iw ][ idx_ic ] = char_to_int_intent[c]
        
        for idx_iw, w in enumerate(o_words):
            for idx_ic, c in enumerate(w):
                O_WORDS[ idx_iw ][ idx_ic ] = char_to_int_intent[c]
        
        
        for idx_iw, w in enumerate(o_programs):
            for idx_ic, c in enumerate(w):
                O_PROGRAMS[ idx_iw ][ idx_ic ] = o_programs[ idx_iw ][ idx_ic ]
        
        yield [I_WORDS, O_WORDS, I_PROGRAMS], O_PROGRAMS

if __name__ == "__main__":
    tokenizer_program = pt.get_tokenizer()
    char_to_int_program = json.loads(tokenizer_program.get_config()['word_index'])
    tam_program_vocabulary = len(tokenizer_program.word_index) + 1
    tokenizer_io = getIOtokenizer()
    char_to_int_intent = json.loads(tokenizer_io.get_config()['word_index'])
    tam_intent_vocabulary = len(tokenizer_io.word_index) + 1
    model = nn.generate_model(tam_intent_vocabulary, tam_program_vocabulary)
    
    fails = True
    input = "hola6 67 k pasa"
    INPUT = np.array(tokenizer_io.texts_to_sequences([list(input)]))
    while fails:
        try:
            program = pg.gen_p()
            output = pr.decode_p(program, input)
            fails = False
        except:
            print("Execution failure, generating a new program")
    OUTPUT = np.array(tokenizer_io.texts_to_sequences([list(output)]))
    translated_program = pt.JSON2RNN([program])[0]
    
    INPUT_P = np.array([[tokenizer_program.word_index['<SOP>']] + translated_program])
    OUTPUT_P = tf.one_hot(np.array([translated_program + [tokenizer_program.word_index['<EOP>']]]), tam_program_vocabulary)
    
    model.fit(x=[INPUT, OUTPUT, INPUT_P], y=OUTPUT_P, epochs=3)
    
    print(next(generator(3)))