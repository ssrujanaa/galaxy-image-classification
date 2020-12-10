#!/usr/bin/env python3
# coding: utf-8

from skimage.transform import resize
import matplotlib.image as mpimg
import os
from glob import glob         
import matplotlib.pyplot as plt

def crop_center(img):
    y,x = img.shape[0],img.shape[1]
    startx = x//2-(64//2)
    starty = y//2-(64//2)    
    return img[starty:starty+64,startx:startx+64]

def main():
    ORIG_SHAPE = (424,424)
    IMG_SHAPE = (64,64)
    dataset = []
    for i in range(8):
        dataset.append('Class0_'+ str(i)+'.jpg')
    for i in range(8):
        dataset.append('Class1_'+ str(i)+'.jpg')
    for i in range(3):
        dataset.append('Class2_'+ str(i)+'.jpg')
    for i in range(5):
        dataset.append('Class3_'+ str(i)+'.jpg')
    for i in range(6):
        dataset.append('Class4_'+ str(i)+'.jpg')
    
    for file in dataset:
        img = plt.imread(file)
        c = crop_center(img)
        name = os.path.splitext(file)[0]
        mpimg.imsave('resized_' + name + '.jpg', img)
    return 0
    
if __name__ == '__main__':
    main()