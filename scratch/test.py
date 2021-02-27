import tensorflow as tf
import numpy as np
import pandas as pd
import random
from glob import glob

from tensorflow.python.keras import backend as K

# tf.enable_eager_execution()
images = glob("input/Class*_*.jpg")
label = [3,4]

def get_images(data_dir,label_dir):
    
    img_width=random.randint(170,241)
    img_height=img_width

    label_list=label_dir
    image_list=data_dir

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
# image = tf.convert_to_tensor(image, dtype=tf.string)
# label = tf.convert_to_tensor(label, dtype=tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    #input_queue = tf.data.Dataset.from_tensor_slices([image,label])

    label = input_queue[1]
    label = tf.cast(label, tf.int32)

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)              

    image = tf.cast(image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, img_width,  img_height)
    image = tf.image.resize_images(image, [80, 80],method=tf.image.ResizeMethod.BILINEAR) 


    image = tf.cast(image, tf.float32)
    return image,label
    
    
image,label=get_images(images,label)
image = tf.random_crop(image, [64, 64, 3])# randomly crop the image size to 45 x 45
image=tf.image.rot90(image,k=random.randint(0,3))
image = tf.image.random_flip_left_right(image)
    
    
    
image = tf.image.random_brightness(image, max_delta=63)
#    image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
#    image = tf.image.random_hue(image, max_delta=0.2)
image = tf.image.random_contrast(image,lower=0.2,upper=1.8)    


#归一化
image = tf.image.per_image_standardization(image)   #substract off the mean and divide by the variance 


    
images, label_batch = tf.train.shuffle_batch(
                                [image, label], 
                                batch_size = 1,
                                num_threads= 64,
                                capacity = 20000,
                                min_after_dequeue = 3000)
images = tf.cast(images, tf.float32)

sess = K.get_session()
# array = sess.run(images)

print(images.eval(session = sess))