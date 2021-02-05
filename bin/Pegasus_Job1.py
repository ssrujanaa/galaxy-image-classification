#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import os       
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
import argparse

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

def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row
        
def label_dataset(csv):        
    df = pd.read_csv(csv)
    df.drop(['Class1.3','Class1.3','Class3.1','Class3.2','Class4.2','Class5.1','Class5.2','Class5.3','Class5.4', 
            'Class6.1','Class6.2','Class8.1','Class8.2','Class8.3','Class8.4','Class8.5','Class8.6','Class8.7',
            'Class9.1','Class9.2','Class9.3','Class10.1','Class10.2','Class10.3','Class11.1','Class11.2','Class11.3',
            'Class11.4','Class11.5','Class11.6'],axis=1, inplace=True)

    new_df = pd.DataFrame(columns = ['GalaxyID', 'Label'])

    for i in range(len(df)):
        label = '$'
        if df.at[i,'Class1.1'] >= 0.469 and df.at[i,'Class7.1'] >= 0.50:
            label = '0'
        elif df.at[i,'Class1.1'] >= 0.469 and df.at[i,'Class7.2'] >= 0.50:
            label = '1'
        elif df.at[i,'Class1.1'] >= 0.469 and df.at[i,'Class7.3'] >= 0.50:
            label = '2'
        elif df.at[i,'Class1.2'] >= 0.430 and df.at[i,'Class2.1'] >= 0.602:
            label = '3'
        elif df.at[i,'Class1.2'] >= 0.469 and df.at[i,'Class2.2'] >= 0.715 and df.at[i,'Class4.1'] >= 0.619:
            label = '4'
        else:
            continue
        if label != '$':
            insert(new_df,[df.at[i,'GalaxyID'], label])
        else:
            continue
            
    return new_df

def get_current_data(csv,input_files):
    df = label_dataset(csv)
    column_names = ["image","label"]
    datafr = pd.DataFrame(columns = column_names)
    
    for i in range(len(df)):
        if df['Label'].iloc[i] == '0' and (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
            datafr.loc[i] = (str(df['GalaxyID'].iloc[i]) +'.jpg'),str(0)

        elif df['Label'].iloc[i] == '1' and (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
            datafr.loc[i] = (str(df['GalaxyID'].iloc[i]) +'.jpg'),str(1)

        elif df['Label'].iloc[i] == '2' and (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
            datafr.loc[i] = (str(df['GalaxyID'].iloc[i]) +'.jpg'),str(2)

        elif df['Label'].iloc[i] == '3' and (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
            datafr.loc[i] = (str(df['GalaxyID'].iloc[i]) +'.jpg'),str(3)

        elif df['Label'].iloc[i] == '4' and (str(df['GalaxyID'].iloc[i]) +'.jpg') in input_files:
            datafr.loc[i] = (str(df['GalaxyID'].iloc[i]) +'.jpg'),str(4)

    return datafr

if __name__ == '__main__':
    
    args = parse_args(sys.argv[1:])
    input_path = args.input_dir
    
    f = os.listdir(input_path)
    input_files = [i for i in f if ".jpg" in i]
    csv = 'training_solutions_rev1.csv'
    
    output_path = args.output_dir
    
    final_df = get_current_data(csv,input_files)
    
    X = final_df.drop(['label'], axis=1)
    y = final_df['label']
    
    X_train1, X_test, y_train1, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
    X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.2, random_state=42, stratify = y_train1)
    
    #Once I split data into train/val/test, I need to rename files and have a csv file that matches its label.
    #Let's first start with training data: X_train and y_train
    
    column_names = ["image","label"]
    train_datafr = pd.DataFrame(columns = column_names)
    
    for i in range(len(X_train)):
        img = plt.imread(X_train.iloc[i][0])
        mpimg.imsave(os.path.join(args.output_dir,'train_' + str(i) + '.jpg'), img)
        train_datafr.loc[i] = ('train_' + str(i) + '.jpg'),  y_train.iloc[i][0]
        
    train_datafr.to_csv(os.path.join(args.output_dir,'Training_Data.csv'), index = False)
    
    #Now, let's save amm images in the validation set
    column_names = ["image","label"]
    val_datafr = pd.DataFrame(columns = column_names)
    
    for i in range(len(X_val)):
        img = plt.imread(X_val.iloc[i][0])
        mpimg.imsave(os.path.join(args.output_dir,'val_' + str(i) + '.jpg'), img)
        val_datafr.loc[i] = ('val_' + str(i) + '.jpg'),  y_val.iloc[i][0]
        
    val_datafr.to_csv(os.path.join(args.output_dir,'Val_Data.csv'), index = False)
    
    #Finally, the test data set
    column_names = ["image","label"]
    test_datafr = pd.DataFrame(columns = column_names)
    
    for i in range(len(X_test)):
        img = plt.imread(X_test.iloc[i][0])
        mpimg.imsave(os.path.join(args.output_dir,'test_' + str(i) + '.jpg'), img)
        test_datafr.loc[i] = ('test_' + str(i) + '.jpg'),  y_test.iloc[i][0]
        
    test_datafr.to_csv(os.path.join(args.output_dir,'Test_Data.csv'), index = False)
    