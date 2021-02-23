# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:07:05 2020

@author: charl
"""
import tensorflow as tf
import numpy as np

def generate_model(tam_intent_vocabulary, tam_program_vocabulary):
   
    input_i   =   tf.keras.layers.Input(shape=(None,))
    embed_i   =   tf.keras.layers.Embedding(input_dim=tam_intent_vocabulary, output_dim=32)(input_i)
    lstm_i_output, lstm_i_h, lstm_i_c    =   tf.keras.layers.LSTM(64, return_state=True)(embed_i)
    
    
    input_o   =   tf.keras.layers.Input(shape=(None,))
    embed_o   =   tf.keras.layers.Embedding(input_dim=tam_intent_vocabulary, output_dim=32)(input_o)
    lstm_o_output, lstm_o_h, lstm_o_c     =   tf.keras.layers.LSTM(64, return_state=True)(embed_o,initial_state=[lstm_i_h, lstm_i_c])
    
    
    input_p   =   tf.keras.layers.Input(shape=(None,))
    embed_p   =   tf.keras.layers.Embedding(input_dim=tam_program_vocabulary, output_dim=32)(input_p)
    lstm_p_output  =   tf.keras.layers.LSTM(64, return_sequences = True)(embed_p,initial_state=[lstm_o_h, lstm_o_c])
    
    # lstm_p_output.shape == [batch_size, longitud de la secuencia del programa, 64]
    
    token_classifier = tf.keras.layers.Dense(tam_program_vocabulary, activation='softmax')(lstm_p_output)
    
    # lstm_p_output.shape == [batch_size, longitud de la secuencia del programa, tam_program_vocabulary] 
    
    model = tf.keras.Model(inputs=[input_i,input_o, input_p], outputs=token_classifier)
    
    model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()])
    return model


def generate_attention_model(tam_intent_vocabulary, tam_program_vocabulary):
    
    input_i   =   tf.keras.layers.Input(shape=(None,))
    embed_i   =   tf.keras.layers.Embedding(input_dim=tam_intent_vocabulary, output_dim=64)(input_i)
    lstm_i_output, lstm_i_h, lstm_i_c    =   tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)(embed_i)
    
    # lstm_i_ouput.shape == [batch_size, longitud secuencia input, 64]
    
    input_o   =   tf.keras.layers.Input(shape=(None,))
    embed_o   =   tf.keras.layers.Embedding(input_dim=tam_intent_vocabulary, output_dim=64)(input_o)
    attention_o2i = tf.keras.layers.Attention()([embed_o, lstm_i_output])
    
    # attention_o2i.shape == [batch_size, longitud secuencia output, 64]
   
    concat_o = tf.keras.layers.Concatenate()([embed_o, attention_o2i])
    lstm_o_output, lstm_o_h, lstm_o_c     =   tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)(concat_o, initial_state=[lstm_i_h, lstm_i_c])
    
    # lstm_o_ouput.shape == [batch_size, longitud secuencia output, 64]
    
    input_p   =   tf.keras.layers.Input(shape=(None,))
    embed_p   =   tf.keras.layers.Embedding(input_dim=tam_program_vocabulary, output_dim=64)(input_p)
    
    # DOUBLE ATTENTION
    
    attention_p2o = tf.keras.layers.Attention()([embed_p, lstm_o_output])
    attention_p2i = tf.keras.layers.Attention()([attention_p2o, lstm_i_output])
    concat_p = tf.keras.layers.Concatenate()([embed_p, attention_p2i])
    lstm_p_output =   tf.keras.layers.LSTM(64, return_sequences=True)(concat_p, initial_state=[lstm_o_h, lstm_o_c])
    
    # lstm_p_output.shape == [batch_size, longitud secuencia program, 64]
    
    token_classifier = tf.keras.layers.Dense(tam_program_vocabulary, activation='softmax')(lstm_p_output)
    model = tf.keras.Model(inputs=[input_i, input_o, input_p], outputs=token_classifier)
    model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()])
    return model

if __name__ == "__main__":
    
    tam_intent_vocabulary = 10
    tam_program_vocabulary = 5
    
    normal_model = generate_model(tam_intent_vocabulary, tam_program_vocabulary)
    
    #normal_model.summary()
    #tf.keras.utils.plot_model(normal_model)
    
    random_input_i = np.zeros((1, 10)) # [batch_size, seq_length de la entrada del intent]
    random_input_o = np.zeros((1, 20)) # [batch_size, seq_length de la salida del intent]
    
    random_input_p_input = np.zeros((1, 1+6))  # [batch_size, <sop> + seq_length del programa]
    random_input_p_prediction = np.zeros((1, 6+1, tam_program_vocabulary)) # [batch_size, seq_length del programa + <eop>, tam_program_vocabulary]
    
    normal_model.fit(x=[random_input_i, random_input_o, random_input_p_input], y= random_input_p_prediction, epochs=3)
    
    tam_intent_vocabulary = 10
    tam_program_vocabulary = 50
    attention_model = generate_attention_model(tam_intent_vocabulary, tam_program_vocabulary)
    
    #attention_model.summary()
    #tf.keras.utils.plot_model(attention_model)
    random_input_i = np.zeros((1, 10)) # [batch_size, seq_length de la entrada del intent]
    random_input_o = np.zeros((1, 20)) # [batch_size, seq_length de la salida del intent]
    
    random_input_p_input = np.zeros((1, 1+6))  # [batch_size, <sop> + seq_length del programa]
    random_input_p_prediction = np.zeros((1, 6+1, tam_program_vocabulary)) # [batch_size, seq_length del programa + <eop>, tam_program_vocabulary]
    
    attention_model.fit(x=[random_input_i, random_input_o, random_input_p_input], y= random_input_p_prediction, epochs=3)
    
