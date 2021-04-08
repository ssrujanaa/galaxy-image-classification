#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob
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
        elif df.at[i,'Class1.2'] >= 0.430 and df.at[i,'Class2.2'] >= 0.715 and df.at[i,'Class4.1'] >= 0.69:
            label = '4'
        else:
            continue
        if label != '$':
            name = 'Class' + label +'_'+ str(df.at[i,'GalaxyID'])
            insert(new_df,[name, label])
        else:
    return new_df

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    input_path = args.input_dir

    f = os.listdir(input_path)
    input_files = glob('/local-work/srujana/galaxy-image-classification2/images_training_rev1/*.jpg')
    df = label_dataset('/local-work/srujana/galaxy-image-classification2/training_solutions_rev1.csv')
    df.to_csv('/local-work/srujana/galaxy-image-classification2/Dataset.csv', index = False)
    output_path = '/local-work/srujana/galaxy-image-classification2/output/'

    count1 =0
    count2 =0
    count3 =0
    count4 =0
    count5 =0

    for i in range(len(df)):
        name = df.at[i,'GalaxyID'].split('_')[1]
        if df['Label'].iloc[i] == '0':
            if ('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name +'.jpg') in input_files:
                img = plt.imread('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+name+'.jpg')
            mpimg.imsave(os.path.join(output_path,'Class0_' + str(df.at[i,'GalaxyID']) + '.jpg'), img)
            count1+=1
        elif df['Label'].iloc[i] == '1':
            if ('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name +'.jpg') in input_files:
                img = plt.imread('/local-work/srujana/galaxy-image-classification2/images_training_rev1/' + name +'.jpg')
            mpimg.imsave(os.path.join(output_path,'Class1_' + str(df.at[i,'GalaxyID']) + '.jpg'), img)
            count2+=1
        elif df['Label'].iloc[i] == '2':
            if ('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name  +'.jpg') in input_files:
                img = plt.imread('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name +'.jpg')
            mpimg.imsave(os.path.join(output_path,'Class2_' + str(df.at[i,'GalaxyID']) + '.jpg'), img)
            count3+=1
        elif df['Label'].iloc[i] == '3':
            if ('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name  +'.jpg') in input_files:
                img = plt.imread('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name +'.jpg')
            mpimg.imsave(os.path.join(output_path,'Class3_' + str(df.at[i,'GalaxyID']) + '.jpg'), img)
            count4+=1
        elif df['Label'].iloc[i] == '4':
            if ('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name +'.jpg') in input_files:
                img = plt.imread('/local-work/srujana/galaxy-image-classification2/images_training_rev1/'+ name +'.jpg')
            mpimg.imsave(os.path.join(output_path,'Class4_' + str(df.at[i,'GalaxyID']) + '.jpg'), img)
            count5+=1

