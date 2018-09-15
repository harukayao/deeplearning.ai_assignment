#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 20:58:39 2018

@author: haruka
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from dnn_app_utils_v2 import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
'''
index = 121
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
'''

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
'''
print('Number of training examples: ' + str(m_train))
print('Number of testing examples: ' + str(m_test))
print('Each image is of size: (' + str(num_px) + ',' + str(num_px) + ',3)')
print('train_x_orig shape: ' + str(train_x_orig.shape))
print('train_y shape: ' + str(train_y.shape))
print('test_x_orig shape: ' + str(test_x_orig.shape))
print('test_y shape: ' + str(test_y.shape))
'''

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

# Standardize data to have feature values between 0 and 1
train_x = train_x_flatten/255
test_x = test_x_flatten/255
'''
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
'''

'''
Bulid the model:
1. Initialize parameters / Define hyperparameters
2. Loop for num_iterations:
    a. Forward propagation
    b. Compute cost function
    c. Backward propagation
    d. Update parameters (using parameters, and grads from backprop) 
4. Use trained parameters to predict labels
'''

#Two-layer neural network
'''
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
'''

### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000,print_cost = True):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    n_x,n_h,n_y = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Loop (gradient descent)
    for i in range(num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. 
        #Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X,W1,b1,'relu')
        A2, cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')
        
        # Compute cost
        cost = compute_cost(A2,Y)
        
        #Initializing backward propagation
        dA2 = -(np.divide(Y,A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. 
        #Inputs: "dA2, cache2, cache1". 
        #Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)
        
        #Retrieve W1,b1,W2,b2 from parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        
        #Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print('Cost after iteration {} : {}'.format(i,np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per tens)')
    plt.title('Learing rate = ' + str(learning_rate))
    plt.show()
    
    return parameters

#parameters = two_layer_model(train_x,train_y,layers_dims=(n_x,n_h,n_y),num_iterations=2500)
#prediction_train = predict(train_x,train_y,parameters)
#prediction_test = predict(test_x,test_y,parameters)
        
#L-layer Neural Network
'''
def initialize_parameters_deep(layer_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
'''
               
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = True):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    costs = []
    
    #Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)
    
    #Loop (gradient descent)
    for i in range(num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X,parameters)
        
        #Computer cost
        cost = compute_cost(AL,Y)
        
        #Backward propagation
        grads = L_model_backward(AL,Y,caches)
        
        #Update parameters
        parameters = update_parameters(parameters,grads,learning_rate)
        
        #Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i,np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    #plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per tens)')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()
    
    return parameters

#parameters = L_layer_model(train_x, train_y, layers_dims,num_iterations=2500)
#pred_train = predict(train_x,train_y,parameters)
#pred_test = predict(test_x,test_y,parameters)
        
my_image = "my_image.jpg"
my_label_y = [0] # the true class of your image (1 -> cat, 0 -> non-cat)


fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")        
        
    
    
    

    
    
    