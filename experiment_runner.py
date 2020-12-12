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

print(tf.version.VERSION)

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
                    o_word = pr.decode_p(o_program, ''.join(i_word))
                    
                    fails = False
                except:
                    #print("Execution failure, generating a new program")
                    pass
            i_words.append(i_word)
            o_words.append(list(o_word))
            translated_program = pt.JSON2RNN([o_program])[0]
            o_programs.append(translated_program + [char_to_int_program['<EOP>']])
            
            
        max_longitud_iwords = max([len(word) for word in i_words])
        max_longitud_owords = max([len(word) for word in o_words])
        
        max_longitud_oprograms = max([len(word) for word in o_programs])
        
        
        
        I_WORDS = np.zeros((tam_lote, max_longitud_iwords), dtype=np.int32)
        O_WORDS = np.zeros((tam_lote, max_longitud_owords), dtype=np.int32)
        I_PROGRAMS = np.zeros((tam_lote, max_longitud_oprograms), dtype=np.int32)
        
        O_PROGRAMS = np.zeros((tam_lote, max_longitud_oprograms, tam_program_vocabulary), dtype=np.int32)
        
        for idx_iw, w in enumerate(i_words):
            for idx_c, c in enumerate(w):
                I_WORDS[ idx_iw ][ idx_c ] = char_to_int_intent[c]
        
        for idx_ow, w in enumerate(o_words):
            for idx_c, c in enumerate(w):
                O_WORDS[ idx_ow ][ idx_c ] = char_to_int_intent[c]
        
        for idx_ip, p in enumerate(I_PROGRAMS):
            I_PROGRAMS[ idx_ip ][ 0 ] = char_to_int_program['<SOP>']
        
        for idx_op, p in enumerate(o_programs):
            for idx_c, c in enumerate(p):
                O_PROGRAMS[ idx_op ][ idx_c ][ o_programs[ idx_op ][ idx_c ] ] = 1
        
        yield [I_WORDS, O_WORDS, I_PROGRAMS], O_PROGRAMS

if __name__ == "__main__":
    tokenizer_program = pt.get_tokenizer()
    char_to_int_program = tokenizer_program.word_index
    tam_program_vocabulary = len(tokenizer_program.word_index) + 1
    
    tokenizer_io = getIOtokenizer()
    char_to_int_intent = tokenizer_io.word_index
    tam_intent_vocabulary = len(tokenizer_io.word_index) + 1
    
    model = nn.generate_model(tam_intent_vocabulary, tam_program_vocabulary)
    
    BATCH_SIZE = 64
    gen_training = generator(BATCH_SIZE)
    gen_validation = generator(BATCH_SIZE)
    
    # fails = True
    # input = "hola6 67 k pasa"
    # INPUT = np.array(tokenizer_io.texts_to_sequences([list(input)]))
    # while fails:
    #     try:
    #         program = pg.gen_p()
    #         output = pr.decode_p(program, input)
    #         fails = False
    #     except:
    #         print("Execution failure, generating a new program")
    # OUTPUT = np.array(tokenizer_io.texts_to_sequences([list(output)]))
    # translated_program = pt.JSON2RNN([program])[0]
    
    # INPUT_P = np.array([[tokenizer_program.word_index['<SOP>']] + translated_program])
    # OUTPUT_P = tf.one_hot(np.array([translated_program + [tokenizer_program.word_index['<EOP>']]]), tam_program_vocabulary)
    
    # model.fit(x=[INPUT, OUTPUT, INPUT_P], y=OUTPUT_P, epochs=3)
    
    history = model.fit(gen_training,
              steps_per_epoch=32,
              validation_data=gen_validation,
              validation_steps=32,
              epochs=3,
              # verbose=2
              )
    
    # print(next(generator(2)))
    