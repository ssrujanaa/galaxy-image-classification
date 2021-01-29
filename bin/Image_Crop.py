#!/usr/bin/env python3
# coding: utf-8

from skimage.transform import resize
import matplotlib.image as mpimg
import sys
import argparse
import os
from glob import glob         
import matplotlib.pyplot as plt

def crop_center(img):
    y,x = img.shape[0],img.shape[1]
    startx = x//2-(64//2)
    starty = y//2-(64//2)    
    return img[starty:starty+64,startx:startx+64]
    
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
    files = os.listdir(input_path)
    
    ORIG_SHAPE = (424,424)
    IMG_SHAPE = (64,64)
    
    dataset = [i for i in files if "Class" in i]
    for i in range(8):
        dataset.append('Class0_'+ str(i)+'.jpg')
    for i in range(8):
        dataset.append('Class1_'+ str(i)+'.jpg')
    for i in range(3):
        dataset.append('Class2_'+ str(i)+'.jpg')
    for i in range(4):
        dataset.append('Class3_'+ str(i)+'.jpg')
    for i in range(6):
        dataset.append('Class4_'+ str(i)+'.jpg')
    
    for file in dataset:
        img = plt.imread(file)
        c = crop_center(img)
        name = os.path.splitext(file)[0]
        mpimg.imsave(os.path.join(args.output_dir,'resized_' + name + '.jpg'), img)
