#!/usr/bin/env python3
# coding: utf-8

from PIL import Image
from glob import glob
import os
import cv2
import matplotlib.image as mpimg


def main():
    path = glob('resized_*_*.jpg')

    for file in path:
        orig_Image = cv2.imread(file)
        rotated_45    = cv2.flip(orig_Image, 0)
        rotated_90 = cv2.flip(orig_Image, 1)
        rotated_180 = cv2.flip(orig_Image, -1)
        name = (os.path.splitext(str(file))[0]).split("_")[1]
        cv2.imwrite('Aug_' + name + '_1.jpg', orig_Image) 
        cv2.imwrite('Aug_' + name + '_2.jpg', rotated_45) 
        cv2.imwrite('Aug_' + name + '_3.jpg', rotated_90) 
        cv2.imwrite('Aug_' + name + '_4.jpg', rotated_180) 
    return 0
    
if __name__ == '__main__':
    main()