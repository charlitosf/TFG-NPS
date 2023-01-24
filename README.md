# TFG-NPS

This is the repository that stores the code associated to the Final Degree Project of [@charlitosf](https://github.com/charlitosf). The related publication can be found at the [Repository of the University of Alicante](http://hdl.handle.net/10045/115985).

## Short description

The goal of this collection of scripts is to generate an interpretable computer program by the means of a neural network that is given a series of input and their related expected output strings. The generated program should be able to transform the provided inputs into the outputs and grasp the underlying pattern of execution so that, given an input not shown to the neural model, the execution of the program passing the input generates the expected output.

The programs are defined using the Domain Specific Language (DSL) described in section 4.3 of the [project report](https://rua.ua.es/dspace/bitstream/10045/115985/1/Synthetic_generation_of_programs_using_Neural_Ne_Ferrus_Ferri_Carlos_Mariano.pdf) (pages 23 and 55).

This repository contains scripts to:
- Generate random programs
- Interpret programs (given program and its input, generate output)
- Train the model
- Run experiments providing different hyperparameters to the neural networks, selecting different models, etc.
- Test the trained model by providing it inputs and outputs so that it generates programs

## Files

- [neural_networks.py](https://github.com/charlitosf/TFG-NPS/blob/master/neural_networks.py): Contains the definition of the Neural Network models used in the project.

- [experiment_runner.py](https://github.com/charlitosf/TFG-NPS/blob/master/experiment_runner.py): Main file used to run experiments based on command-line parameters.

- [program_generator.py](https://github.com/charlitosf/TFG-NPS/blob/master/program_generator.py): Generator of random programs defined using the DSL.

- [program_runner.py](https://github.com/charlitosf/TFG-NPS/blob/master/program_runner.py): Interprets a program coded using the DSL of the project. Given an input to the program and the description of the program itself, generates its output.

- [program_translator.py](https://github.com/charlitosf/TFG-NPS/blob/master/program_translator.py): Translates the definition of a program from the Neural Network encoding to JSON and vice-versa.

- [model_tester.py](https://github.com/charlitosf/TFG-NPS/blob/master/model_tester.py): Loads the trained model and lets the user provide inputs and expected outputs. Then, runs the Neural Network to predict the program that could generalize the transformation of the inputs into the outputs.
