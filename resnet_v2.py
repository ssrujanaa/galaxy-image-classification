from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim

import resnet_utils
# from nets import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
    
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
        
        residual = tf.nn.dropout(residual,1.0)

        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)
    
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                            
                        output_stride /= 4
                        
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        net = resnet_utils.conv2d_same(net, 64, 6, stride=1, scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')     
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)    
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                output0=net
                
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    output1=net
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
                    
                if spatial_squeeze:
                    logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                    
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(logits, scope='predictions')
                    
                return logits, end_points,output0,output1

resnet_v2.default_image_size = 64
def resnet_v2_26_2(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_26_2'):
    k=2
    blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256*k, 64*k, 1)]  + [(256*k, 64*k, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512*k, 128*k, 1)] + [(512*k, 128*k, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024*k, 256*k, 1)] + [(1024*k, 256*k, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048*k, 512*k, 1)] * 2)]
    
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)

resnet_v2_26_2.default_image_size = resnet_v2.default_image_size

def loss(logits, labels):
    with tf.name_scope('Loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.add_to_collection('losses',loss)
        losses=tf.add_n( tf.get_collection('losses'),name='total_loss')
        tf.summary.scalar(scope+'/loss', losses)
        return losses

def compute_rmse(logits, labels):
    with tf.name_scope('RMSE') as scope:
        logits=tf.nn.softmax(logits,dim=-1)
        rmse=tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(labels-logits),reduction_indices=[1])))
        tf.summary.scalar(scope+'/rmse', rmse)
    return rmse

def accuracy(logits, labels):
    with tf.name_scope('Accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy

def num_correct_prediction(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct

def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer')as scope:
        learning_rate = tf.train.exponential_decay(learning_rate, global_step,30000, 0.1, staircase=True)
        tf.summary.scalar(scope+'/learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9,use_nesterov=True)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op