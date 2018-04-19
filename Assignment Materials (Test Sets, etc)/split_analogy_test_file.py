#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:56:55 2018

@author: jackharding
"""

def split_analogy_test_file():
    with open("Analogy-Benchmark.txt", "r") as f_in:
        f_out = None
        lines = f_in.readlines()
        for idx, line in enumerate(lines):
            line_split = line.strip().split()
            if line_split[0] == ':':
                try:
                    f_out.close()
                except:
                    pass
                
                f_out = open('Analogies/{}.txt'.format(line_split[1]), 'w')
                continue

            print(line, file=f_out, end='')

        if idx == len(lines) - 1:
            f_out.close()
            
split_analogy_test_file()