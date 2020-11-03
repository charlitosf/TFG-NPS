# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:27:17 2020

@author: charl
"""

import argparse
import json

def decode_p(p):
    print(p)

if __name__ == '__main__':
    ## Parsing arguments
    parser = argparse.ArgumentParser(description='Parser for a program described as a JSON')
    parser.add_argument('file')
    args = parser.parse_args()
    
    ## Loading data from input file
    with open(args.file, 'r') as f:
        p = json.load(f)
    
    ## Decoding the program
    decode_p(p)