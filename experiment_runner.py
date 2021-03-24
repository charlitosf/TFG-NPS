# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:28:44 2020

@author: charl
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import neural_networks as nn
import program_generator as pg
import program_runner as pr
import program_translator as pt
import numpy as np
import tensorflow as tf
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import random
#random.seed(1235)

print(f"Tensorflow version: {tf.version.VERSION}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:    
    print("GPU found")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.preprocessing.text import Tokenizer

with open("config.json", 'r') as fp:
    CONFIG = json.load(fp)
    fp.close()

CONFIG = CONFIG['SIMPLIFIED_1']


def getIOtokenizer():
    tokenizer = Tokenizer(filters='',
                      lower=False,
                      #oov_token="<UNK>"
                      )
    tokenizer.fit_on_texts(["<EOP>", "<SOP>"])
    characters = [list(c) for c in CONFIG['c']]
    tokenizer.fit_on_texts(characters)
    return tokenizer

def integer_list2string(integer_list):
    return "".join([int_to_char_intent[tok] for tok in integer_list])

def test_model(model, generator):
    inputs, output = next(generator)
    
    inputs[2] = np.array(list(map(lambda values: values[0] + [0] * CONFIG['MAX_PROGRAM_SIZE'], inputs[2])))
    
    predictions = model(inputs)
    expectations = pt.RNN2JSON(output)
    prediction_chars = pt.rnn2list(predictions)
    
    for i in range(predictions.shape[0]):
        in_str = integer_list2string(inputs[0][i])
        out_str = integer_list2string(inputs[1][i])
        print('\nIntent:')
        print(f'Input: "{in_str}"')
        print(f'Output: "{out_str}"\n')
        
        
        print('Expected output program:')
        print(expectations[i])
        
        print('Actual output program:')
        correct = pt.is_rnn_program_correct(predictions[i])
        if correct:
            json_prediction = pt.RNN2JSON([predictions[i]])[0]
            print(json_prediction)
            distance = pr.check_consistency(json_prediction, in_str, out_str)
            print(f'Actual output: "{pr.decode_p(json_prediction, in_str)}"')
            print(f'Levenshtein distance: {distance}')
        else:
            print(prediction_chars[i])
            print(f'Correct? {correct}')
        print()
            

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

def translate_wordList(wordlist, char_to_int, result):
    for idx, w in enumerate(wordlist):
            for idx_c, c in enumerate(w):
                result[ idx ][ idx_c ] = char_to_int[c]
    return result
    
    
def generator(tam_lote = 32):
    while True:
        i_words = []
        o_words = []
        
        o_programs = []
        i = 0
        while i < tam_lote:
            amount_inputs = random.randint(CONFIG['MIN_INPUTS_PER_PROGRAM'], CONFIG['MAX_INPUTS_PER_PROGRAM'])
            if i + amount_inputs >= tam_lote:
                amount_inputs = tam_lote - i
            
            fails = True
            while fails:
                i_words_program = []
                o_words_program = []
                j = 0
                o_program = pg.gen_p()
                fails = False
                while j < amount_inputs and not fails:
                    try:
                        i_word = pg.gen_word()
                        o_word = list(pr.decode_p(o_program, ''.join(i_word)))
                        if len(o_word) == 0:
                            fails = True
                    except:
                        fails = True
                        #print("Execution failure, generating a new program")
                    if not fails:
                        i_words_program.append(i_word)
                        o_words_program.append(o_word)
                    
                        j += 1
                        
                        
            
            i += j
            i_words += i_words_program
            o_words += o_words_program
            
            translated_program = pt.JSON2RNN([o_program])[0]
            for _ in range(amount_inputs):
                o_programs.append(translated_program + [char_to_int_program['<EOP>']])
            
        if len(i_words) != tam_lote or len(o_words) != tam_lote:
            print(len(i_words), len(o_words))
        max_longitud_iwords = max([len(word) for word in i_words])
        max_longitud_owords = max([len(word) for word in o_words])
        
        max_longitud_oprograms = max([len(word) for word in o_programs])
        
        
        I_WORDS = np.zeros((tam_lote, max_longitud_iwords), dtype=np.int32)
        O_WORDS = np.zeros((tam_lote, max_longitud_owords), dtype=np.int32)
        I_PROGRAMS = np.zeros((tam_lote, max_longitud_oprograms), dtype=np.int32)
        
        O_PROGRAMS = np.zeros((tam_lote, max_longitud_oprograms, tam_program_vocabulary), dtype=np.int32)
        
        I_WORDS = translate_wordList(i_words, char_to_int_intent, I_WORDS)
        
        O_WORDS = translate_wordList(o_words, char_to_int_intent, O_WORDS)
        
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
        
def getParser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--train', action='store_true', help='Set for training the model')
    parser.add_argument('--same_data_for_validation', action='store_true', help='Instead of generating (or using) different data between the training and validation datasets, use the same')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size used for the model')
    parser.add_argument('--evolution_graph', action='store_true', help='Set for printing a graph with the evolution of the loss trhought the training')
    parser.add_argument('--total_epochs', type=int, default=20, help='Total amount of epochs to run when training')
    parser.add_argument('--epochs_per_superepoch', type=int, default=10, help='Amount of epochs for each superepoch (these are actual keras epochs)')
    parser.add_argument('--saving_frequency', type=int, default=10, help='Frequency (in epochs) in which a checkpoint will be saved')
    parser.add_argument('--tests_per_superepoch', type=int, default=1, help='Amount of examples passed to the network for each superepoch (for testing purposes)')
    parser.add_argument('--use_generator', action='store_true', help='Use generator a generator function for training instead of a whole dataset')
    parser.add_argument('--use_attention', action='store_true', help='Generate the model using the attetion mechanism')
    parser.add_argument('--use_last_checkpoint', action='store_true', help='Start the model from the last checkpoint')
    return parser

def getModel(train = True, examples_per_epoch = 4096, validation_ratio = 8, batch_size = 32, total_epochs = 20,
             epochs_per_superepoch = 10, saving_frequency = 10, evolution_graph = True, tests_per_superepoch = 0,
             use_generator = True, same_data_for_validation = False, attention = False, use_last_checkpoint = False):
    
    if attention:
        model = nn.generate_model(tam_intent_vocabulary, tam_program_vocabulary)
    else:
        model = nn.generate_attention_model(tam_intent_vocabulary, tam_program_vocabulary)
    EXAMPLES_PER_EPOCH = examples_per_epoch
    EXAMPLES_PER_EPOCH_VALIDATION = int(EXAMPLES_PER_EPOCH / validation_ratio)
    BATCH_SIZE = batch_size
    VERBOSE = 2
    
    gen_training = generator(BATCH_SIZE)
    gen_validation = generator(BATCH_SIZE)
    
    steps_per_epoch = int(EXAMPLES_PER_EPOCH / BATCH_SIZE)
    validation_steps = int(EXAMPLES_PER_EPOCH_VALIDATION / BATCH_SIZE)
    
    if attention:
        CHECKPOINT_PREFIX = 'att_checkpoints/last_cp'
    else:
        CHECKPOINT_PREFIX = 'checkpoints/last_cp'
    
    if use_last_checkpoint:
        model.load_weights(CHECKPOINT_PREFIX)
    
    if train:
        loss = []
        val_loss = []
        testing_generator = generator(tests_per_superepoch)
        if not use_generator:
            dataset_input, dataset_output = next(generator(steps_per_epoch * BATCH_SIZE))
            
        if same_data_for_validation:
            same_data_generator = generator(steps_per_epoch * BATCH_SIZE)
            
        for superepoch in range(int(total_epochs/epochs_per_superepoch)):
            print(f'\nSuperepoch {superepoch + 1}/{int(total_epochs/epochs_per_superepoch)}:\n')
            if use_generator and not same_data_for_validation:
                history = model.fit(gen_training,
                          steps_per_epoch=steps_per_epoch,
                          validation_data=gen_validation,
                          validation_steps=validation_steps,
                          epochs=epochs_per_superepoch,
                          verbose=VERBOSE,
                          )
                superepoch_loss = history.history['loss']
                superepoch_val_loss = history.history['val_loss']
                
            elif use_generator and same_data_for_validation:
                superepoch_loss = []
                superepoch_val_loss = []
                for _ in range(epochs_per_superepoch):
                    data_in, data_out = next(same_data_generator)
                    history = model.fit(x=data_in,
                              y=data_out,
                              batch_size=BATCH_SIZE,
                              validation_data=(data_in, data_out),
                              verbose=VERBOSE,
                              )
                    superepoch_loss += history.history['loss']
                    superepoch_val_loss += history.history['val_loss']
                    
            elif not same_data_for_validation:
                history = model.fit(x=dataset_input,
                          y=dataset_output,
                          batch_size=BATCH_SIZE,
                          epochs=epochs_per_superepoch,
                          validation_split=0.1,
                          verbose=VERBOSE,
                          )
                superepoch_loss = history.history['loss']
                superepoch_val_loss = history.history['val_loss']
                
            else:
                history = model.fit(x=dataset_input,
                          y=dataset_output,
                          batch_size=BATCH_SIZE,
                          epochs=epochs_per_superepoch,
                          validation_data=(dataset_input, dataset_output),
                          verbose=VERBOSE,
                          )
                superepoch_loss = history.history['loss']
                superepoch_val_loss = history.history['val_loss']
                
            loss += superepoch_loss
            val_loss += superepoch_val_loss
            
            if saving_frequency > 0 and superepoch % saving_frequency == 0:
                model.save_weights(CHECKPOINT_PREFIX)
            
            if evolution_graph:
                epochs_list = range(1, (superepoch + 1) * epochs_per_superepoch + 1)
                fix, ax = plt.subplots()
                ax.plot(epochs_list, val_loss, color='orange', label='Validation Loss')
                ax.plot(epochs_list, loss, color='blue', label='Loss')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend()
                ax.set_title("Evolution of epochs")
                plt.show()
            
            if (tests_per_superepoch > 0):
                test_model(model, testing_generator)
    
    return model

def get_model_from_args(args):
    return getModel(
            args.train,
            examples_per_epoch=2 ** 15,
            batch_size=args.batch_size,
            total_epochs=args.total_epochs,
            epochs_per_superepoch=args.epochs_per_superepoch,
            evolution_graph=args.evolution_graph,
            saving_frequency=args.saving_frequency,
            tests_per_superepoch=args.tests_per_superepoch,
            use_generator=args.use_generator,
            same_data_for_validation=args.same_data_for_validation,
            attention=args.use_attention,
            use_last_checkpoint=args.use_last_checkpoint,
        )

tokenizer_program = pt.get_tokenizer()
char_to_int_program = tokenizer_program.word_index
tam_program_vocabulary = len(tokenizer_program.word_index) + 1

tokenizer_io = getIOtokenizer()
char_to_int_intent = tokenizer_io.word_index
tam_intent_vocabulary = len(tokenizer_io.word_index) + 1

if __name__ == "__main__":
    
    parser = getParser("Experiment runner")
    args = parser.parse_args()
    int_to_char_intent = tokenizer_io.index_word
    
    # sys.stdout = open(datetime.today().strftime('%Y-%m-%d_%H-%M-%S.log'), 'w')
    get_model_from_args(args)
    # sys.stdout.close()