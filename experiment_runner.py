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


if __name__ == "__main__":
    tokenizer = pt.get_tokenizer()
    tam_program_vocabulary = len(tokenizer.word_index) + 1
    tam_intent_vocabulary = tam_program_vocabulary #pg.get_intent_vocabulary()
    model = nn.generate_model(tam_intent_vocabulary, tam_program_vocabulary)
    
    fails = True
    input = "hola6 67 k pasa"
    INPUT = np.array(tokenizer.texts_to_sequences([list(input)]))
    while fails:
        try:
            program = pg.gen_p()
            output = pr.decode_p(program, input)
            fails = False
        except:
            print("Execution failure, generating a new program")
    OUTPUT = np.array(tokenizer.texts_to_sequences([list(output)]))
    translated_program = pt.JSON2RNN([program])[0]
    
    INPUT_P = np.array([[tokenizer.word_index['<SOP>']] + translated_program])
    OUTPUT_P = tf.one_hot(np.array([translated_program + [tokenizer.word_index['<EOP>']]]), tam_program_vocabulary)
    
    model.fit(x=[INPUT, OUTPUT, INPUT_P], y=OUTPUT_P, epochs=3)