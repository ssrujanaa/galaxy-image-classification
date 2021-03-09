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

def get_images(data_dir,label_dir):
    img_width=220
    img_height=img_width
    numpy_image = []
    numpy_label = []
    
    label_list=label_dir
    image_list=data_dir
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [round(float(i)) for i in label_list] 

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    # make an input queue
    input_queue = tf.data.Dataset.from_tensor_slices((image,label))
    i=0
    for ele in list(iter(input_queue)):
        next_item = ele
        print(i,next_item)
        i=i+1

        label = next_item[1]
        label = tf.cast(label, tf.int32)

        image_contents = tf.io.read_file(next_item[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)              

        image = tf.cast(image, tf.float32)

        image = tf.image.resize_with_crop_or_pad(image, img_width,  img_height)
        image = tf.image.resize(image, [80, 80],method=tf.image.ResizeMethod.BILINEAR) 
        image = tf.image.resize_with_crop_or_pad(image, 64,  64)
        image = tf.image.per_image_standardization(image)

        image = tf.cast(image, tf.float32)

        n_classes = 5
        label = tf.one_hot(label, depth= n_classes)
        label = tf.cast(label, dtype=tf.int32)

        numpy_image.append(image.numpy())
        numpy_label.append(label.numpy())

    return numpy_image, numpy_label

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    input_path = args.input_dir
    output_path = args.output_dir
    
    f = os.listdir(input_path)
    images = [i for i in f if ".jpg" in i]
    label = []
    for i in range(len(images)):
        if "Class0" in images[i]:
            label.append(0)
        elif "Class1" in images[i]:
            label.append(1)
        elif "Class2" in images[i]:
            label.append(2)
        elif "Class3" in images[i]:
            label.append(3)
        else:
            label.append(4)
    
    numpy_image, numpy_label = get_images(images,label)
    image_file = "Val_images_Preprocess.npy"
    label_file = "Val_labels_Preprocess.npy"
    np.save(image_file, numpy_image)
    np.save(label_file,numpy_label)

