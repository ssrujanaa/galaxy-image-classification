#!/usr/bin/env python3
# coding: utf-8

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from glob import glob
import sys
import argparse
import os

def parse_args(args):
    parser = argparse.ArgumentParser(description="Enter description here")
    parser.add_argument(
                "-i",
                "--input_dir",
                default=".",
                help="directory where input files will be read from"
            )

    parser.add_argument(
                "-o",
                "--output_dir",
                default=".",
                help="directory where output files will be written to"
            )

    return parser.parse_args(args)
    
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    input_path = args.input_dir
    
  
    f = os.listdir(input_path)
    images = [i for i in f if "resized" in i]
    
    column_names = ["image","label"]
    datafr = pd.DataFrame(columns = column_names)

    for i in range(len(images)):
        if 'Class0' in images[i]:
            label = 0
        elif 'Class1' in images[i]: 
            label = 1
        elif 'Class2' in images[i]:
            label = 2
        elif 'Class3' in images[i]:
            label = 3
        elif 'Class4' in images[i]:
            label = 4

        datafr.loc[i] = images[i],str(label)

    X = datafr.drop(['label'], axis=1)
    y = datafr['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify = y)

    os =  RandomOverSampler(sampling_strategy='all')
    X_new, y_new = os.fit_sample(X, y)

    train_datafr = pd.DataFrame(columns = column_names)
    for i in range(len(X_new)):
        train_datafr.loc[i] = X_new.iloc[i][0], y_new.iloc[i][0]
    train_datafr.to_csv('Training_Data.csv', index = False)

    test_datafr = pd.DataFrame(columns = column_names)
    for i in range(len(X_test)):
        test_datafr.loc[i] = X_test.iloc[i][0], y_test.iloc[i][0]   
    test_datafr.to_csv('Test_Data.csv',index = False)