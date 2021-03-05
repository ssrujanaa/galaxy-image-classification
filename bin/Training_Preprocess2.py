#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import random
import pandas as pd
from glob import glob
from pathlib import Path
import sys
import os
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

RANDOM_CROP_FACTOR = 2  #for dev purposes

def read_galaxy(images, labels):
    random_crop = []
    image_rot = []
    random_flip = []
    final = []
    label_list1 =[]
    label_list2=[]
    numpy_dict = {}
    
    #data augmentation
    for j in range(len(images)):
        for i in range(RANDOM_CROP_FACTOR):
            im = tf.convert_to_tensor(images[i], dtype=tf.float32)
            random_crop.append(tf.image.random_crop(images[i], [64, 64, 3]))
            label_list1.append(labels[j])

    for i in range(len(random_crop)):
        for j in range(4):
            image_rot.append(tf.image.rot90(random_crop[i],k=random.randint(0,3)))
            label_list2.append(label_list1[i])

    label_list1 =[]
    for i in range(len(image_rot)):
        for j in range(2):
            random_flip.append(tf.image.random_flip_left_right(image_rot[i]))
            label = tf.cast(label_list2[i], tf.int32)
            label_list1.append(label.numpy())

    for i in range(len(random_flip)):    
        im = tf.image.random_brightness(random_flip[i], max_delta=63)
        im = tf.image.random_contrast(im,lower=0.2,upper=1.8) 
        #im = tf.image.per_image_standardization(im) 
        final.append(im.numpy())
        
    return final, label_list1

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    input_path = args.input_dir
    output_path = args.output_dir
    
    f = os.listdir(input_path)
    input_images = [i for i in f if "Training_images_Preprocess1.npy" in i]
    input_labels = [i for i in f if "Training_labels_Preprocess1.npy" in i]
    
    images = np.load(input_images[0], allow_pickle=True)
    labels = np.load(input_labels[0], allow_pickle=True)
    numpy_image, numpy_label = read_galaxy(images,labels)
    
    image_file = "Training_images_Preprocess2.npy"
    label_file = "Training_labels_Preprocess2.npy"
    np.save(image_file, numpy_image)
    np.save(label_file,numpy_label)
