from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import sys

activations = {"relu" : tf.nn.relu, "identity" : tf.identity, "lrelu" : tf.nn.leaky_relu}

def generic_cnn(input, model_params):

        print("INPUT NODE : "+str(input.name))
        
        nb_conv_layers = model_params["nb_conv_layers"]
        nb_fc_layers = model_params["nb_fc_layers"]
        
        conv_f = model_params["conv_f"] # features
        conv_k = model_params["conv_k"] # kernels
        conv_p = model_params["conv_p"] # poolings
        conv_a = activations[model_params["conv_a"]] # activation
        
        fc_l = model_params["fc_l"] # layers
        fc_a = activations[model_params["fc_a"]] # activation

        nb_outputs = model_params["nb_outputs"]
        
        prev_input_conv = input
        
        with tf.name_scope("generic_cnn"):
                with tf.name_scope("convolutions") as scope:
                        
                        for i in range(0, nb_conv_layers):
                                with tf.variable_scope("conv"+str(i)+"_0") as scope:
                                        net = tf.contrib.layers.conv2d(prev_input_conv, conv_f[i], conv_k[i], activation_fn=conv_a,
                                                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                                            scope=scope)
                                # with tf.variable_scope("conv"+str(i)+"_1") as scope:
                                #         net = tf.contrib.layers.conv2d(net, conv_f[i], conv_k[i], activation_fn=conv_a,
                                #                                             weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                #                                             scope=scope)

                                net = tf.contrib.layers.max_pool2d(net, conv_p[i], padding='VALID')
                                prev_input_conv = net
                                
                        with tf.variable_scope("output") as scope:
                                net_shape = net.get_shape().as_list()
                                net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
                                # net = tf.contrib.layers.flatten(net)

                prev_input_fc = net
                                
                with tf.name_scope("fully_connected") as scope:

                        for i in range(0, nb_fc_layers):
                                with tf.name_scope("fc"+str(i)) as scope:
                                        fc_net = tf.contrib.layers.fully_connected(prev_input_fc, fc_l[i], activation_fn=fc_a,
                                                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                                   scope=scope)
                                        prev_input_fc = fc_net

                        with tf.name_scope("output") as scope:
                                train_fc_net = tf.contrib.layers.fully_connected(fc_net, nb_outputs, activation_fn=tf.identity,
                                                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                           scope=scope)
                        with tf.name_scope("inference_output") as scope:
                                inference_fc_net = tf.nn.softmax(train_fc_net)
                                # inference_fc_net = tf.contrib.layers.fully_connected(fc_net, nb_outputs, activation_fn=tf.nn.softmax,
                                #                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                #                                            scope=scope)
                        
        print("OUTPUT NODE : "+str(train_fc_net.name))
        print("INFERENCE OUTPUT NODE : "+str(inference_fc_net.name))
        print(inference_fc_net)
        return train_fc_net
