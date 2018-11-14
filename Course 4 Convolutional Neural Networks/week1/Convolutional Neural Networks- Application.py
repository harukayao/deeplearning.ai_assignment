#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 00:07:13 2018

@author: haruka
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
'''
index = 6
plt.imshow(X_train_orig[index])
print('y = ' + str(np.squeeze(Y_train_orig[:, index])))
'''

'''
X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print('number of training examples = ' + str(X_train.shape[0]))
print('number of test examples = ' + str(X_test.shape[0]))
print('X_train shape: ' + str(X_train.shape))
print('Y_train shape: ' + str(Y_train.shape))
print('X_test shape: ' + str(X_test.shape))
print('Y_test shape: ' + str(Y_test.shape))
conv_layers = {}
'''

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(name = 'X', shape = (None, n_H0, n_W0, n_C0), dtype = tf.float32)
    Y = tf.placeholder(name = 'Y', shape = (None, n_y), dtype=tf.float32)
    
    return X, Y

'''
X, Y = create_placeholders(64,64,3,6)
print('X = ' + str(X))
print('Y = ' + str(Y))
'''

def initialize_parameter():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    tf.set_random_seed(1)
    
    W1 = tf.get_variable(name = 'W1', dtype=tf.float32, shape=(4,4,3,8), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable(name = 'W2', dtype=tf.float32, shape=(2,2,8,16), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {'W1': W1,
                  'W2': W2}
    return parameters
'''
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameter()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print('W1 = ' + str(parameters['W1'].eval()[1,1,1]))
    print('W2 = ' + str(parameters['W2'].eval()[1,1,1]))
'''

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    #CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(input=X, filter=W1, strides=(1,1,1,1), padding='SAME')
    #RELU
    A1 = tf.nn.relu(Z1)
    #MAXPOOL: window 8*8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(value=A1, ksize=(1,8,8,1), strides=(1,8,8,1), padding='SAME')
    #CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=(1,1,1,1), padding='SAME')
    #RELU
    A2 = tf.nn.relu(Z2)
    #MAXPOOL: window 4*4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(value=A2, ksize=(1,4,4,1), strides=(1,4,4,1), padding='SAME')
    #FLATTEN
    P2 = tf.contrib.layers.flatten(inputs=P2)
    # FULLY-CONNECTED without non-linear activation function (not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    return Z3

tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64,64,3,6)
    parametes = initialize_parameter()
    Z3 = forward_propagation(X, parametes)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3,{X:np.random.randn(2,64,64,3),
                     Y:np.random.randn(2,6)})
    print('Z3 = ' + str(a))
    