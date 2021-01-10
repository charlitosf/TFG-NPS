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
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

print(f"Tensorflow version: {tf.version.VERSION}")
print(f"GPU found: {len(tf.config.experimental.list_physical_devices('GPU')) != 0}")

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

"""
Función generadora de los datos de la red neuronal

Genera lotes de tamaño tam_lote

- Por cada elemento del lote:
  - Genera una string como input
  - Hasta que funcione:
    - Genera un programa
    - Ejecuta el programa para obtener la string de output
  - Traduce el programa finalmente generado (el primero que funcione) a array de enteros y le añade un <EOP> al final
- Genera un lote de strings de entrada del tamaño de la string de entrada más larga generada
- Genera un lote de strings de salida del tamaño de la string de salida más larga generada
- Genera un lote de programas del tamaño del programa más largo generado, traducido a one-hot
- Genera un lote del mismo tamaño que el anterior (sin traducir a one-hot) que empieza con <SOP> y el resto ceros
- Devuelve un vector con los lotes de strings de entrada, salida y el "programa" de entrada (la <SOP> y los ceros), junto al lote de programas de salida
"""
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
        

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument('--train', action='store_true', help='Set for training the model. False for loading it from the last checkpoint')
    args = parser.parse_args()
    
    
    tokenizer_program = pt.get_tokenizer()
    char_to_int_program = tokenizer_program.word_index
    tam_program_vocabulary = len(tokenizer_program.word_index) + 1
    
    tokenizer_io = getIOtokenizer()
    char_to_int_intent = tokenizer_io.word_index
    tam_intent_vocabulary = len(tokenizer_io.word_index) + 1
    
    model = nn.generate_model(tam_intent_vocabulary, tam_program_vocabulary)
    EXAMPLES_PER_EPOCH = 2 ** 15
    EXAMPLES_PER_EPOCH_VALIDATION = int(EXAMPLES_PER_EPOCH / 8)
    BATCH_SIZE = 64
    
    
    
    gen_training = generator(BATCH_SIZE)
    gen_validation = generator(BATCH_SIZE)
    
    steps_per_epoch = int(EXAMPLES_PER_EPOCH / BATCH_SIZE)
    validation_steps = int(EXAMPLES_PER_EPOCH_VALIDATION / BATCH_SIZE)
    
    TRAIN = args.train
    
    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    checkpointdir = os.path.dirname(checkpoint_path)
    
    SAVING_FREQUENCY = 10 # epochs
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq=SAVING_FREQUENCY*steps_per_epoch)
    
    model.save_weights(checkpoint_path.format(epoch=0))
    
    
    if TRAIN:
        EPOCHS = 50
        history = model.fit(gen_training,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=gen_validation,
                  validation_steps=validation_steps,
                  epochs=EPOCHS,
                  # callbacks=[cp_callback]
                  )
        
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_list = range(1, EPOCHS + 1)
        
        fix, ax = plt.subplots()
        ax.plot(epochs_list, loss, label='Loss')
        ax.plot(epochs_list, val_loss, label='Validation Loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.set_title("Evolution of epochs")
    else:
        latest = tf.train.latest_checkpoint(checkpointdir)
        model.load_weights(latest)
    
    
    