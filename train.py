#!/usr/bin/env python3
# coding: utf-8

import os
import os.path
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import resnet_utils
import tf_slim as slim
resnet_arg_scope = resnet_utils.resnet_arg_scope
import resnet_v2

resnet_arg_scope = resnet_utils.resnet_arg_scope

def train():
    N_CLASSES = 5
    BATCH_SIZE =5
    learning_rate = 0.1
    MAX_STEP = 20

    train_images = np.load('./Training_images_Preprocess2.npy', allow_pickle=True)
    train_labels = np.load("./Training_labels_Preprocess2.npy", allow_pickle=True)
    train_tf_images = []
    train_tf_labels = []

    for i in range(len(train_images)):
        train_tf_images.append(tf.convert_to_tensor(train_images[i], dtype=tf.float32))
        train_tf_labels.append(tf.convert_to_tensor(train_labels[i], dtype=tf.int32)) 

    val_images = np.load('./Val_images_Preprocess.npy', allow_pickle=True)
    val_labels = np.load("./Val_labels_Preprocess.npy", allow_pickle=True)
    val_tf_images = []
    val_tf_labels = []

    for i in range(len(val_images)):
        val_tf_images.append(tf.convert_to_tensor(val_images[i], dtype=tf.float32))
        val_tf_labels.append(tf.convert_to_tensor(val_labels[i], dtype=tf.int32)) 

    test_images = np.load('./Test_images_Preprocess.npy', allow_pickle=True)
    test_labels = np.load("./Test_labels_Preprocess.npy", allow_pickle=True)
    test_tf_images = []
    test_tf_labels = []

    for i in range(len(test_images)):
        test_tf_images.append(tf.convert_to_tensor(test_images[i], dtype=tf.float32))
        test_tf_labels.append(tf.convert_to_tensor(test_labels[i], dtype=tf.int32)) 

    def input_fn(train_data, train, batch_size=BATCH_SIZE,buffer_size=1024):
        dx = tf.data.Dataset.from_tensor_slices(train_data[0])
        dy = tf.data.Dataset.from_tensor_slices(train_data[1])
        dataset = tf.data.Dataset.zip((dx,dy))
        if train:
            dataset = dataset.shuffle(buffer_size=buffer_size)
            num_repeat = None
        else:
            num_repeat = 1
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        images_batch, labels_batch = iterator.get_next()
        
        return images_batch,labels_batch

    def train_input_fn():
        return input_fn([train_tf_images,train_tf_labels], train=True)

    def val_input_fn():
        return input_fn([val_tf_images,val_tf_labels], train=True)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3])
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES])

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, end_points,output0,output1 = resnet_v2.resnet_v2_26_2(x, N_CLASSES, is_training=True)
        
    loss = resnet_v2.loss(logits, y_)
    accuracy = resnet_v2.accuracy(logits, y_)

    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = resnet_v2.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()   

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
    tra_summary_writer = tf.summary.FileWriter('./', sess.graph)
    val_summary_writer = tf.summary.FileWriter('./', sess.graph)
        
    try:    
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_image_batch, tra_label_batch = train_input_fn()
            val_image_batch, val_label_batch = val_input_fn()
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _,tra_loss,tra_acc,summary_str = sess.run([train_op,loss,accuracy,summary_op],feed_dict={x:tra_images, y_:tra_labels}) 

            if step % 1 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, tra_loss: %.4f, tra_accuracy: %.2f%%' % (step, tra_loss, tra_acc))
                tra_summary_writer.add_summary(summary_str, step)

            if step % 5 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc, summary_str = sess.run([loss, accuracy,summary_op],feed_dict={x:val_images,y_:val_labels})
                print('**  Step %d, test_loss = %.4f, test_accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                val_summary_writer.add_summary(summary_str, step)

            if step % 10 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join('./', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            
    except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()        

train()

