from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
#use slim api to build the network
def CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 20, 20, 1])

        #conv layer
        net = slim.conv2d(x, 32, [5, 5], scope='conv1')
        #pooling layer
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        #2nd conv layer
        net = slim.conv2d(net, 48, [5, 5], scope='conv2')
        #pooling layer
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #flatten layer: reshape the img into 1 dimension data
        net = slim.flatten(net, scope='flatten3')
        #fully connected layer
        net = slim.fully_connected(net, 1024, scope='fc3')
        #dropout 
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs
