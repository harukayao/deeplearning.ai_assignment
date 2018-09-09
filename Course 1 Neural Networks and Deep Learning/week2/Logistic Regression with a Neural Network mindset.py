#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 19:41:15 2018

@author: haruka
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from skimage import transform
from PIL import Image

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])
    
    classes = np.array(test_dataset['list_classes'][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))
    
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

#loading the data
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()

'''
#Example of a picture
index = 150
plt.imshow(train_set_x_orig[index])
print('y = ' + str(train_set_y[:,index]) + ', it is a ' + 
      classes[np.squeeze(train_set_y[:,index])].decode('utf-8') + ' picture.')
'''

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

'''
print('Number of training examples: m_train = ' + str(m_train))
print('Number of testing examples: m_test = ' + str(m_test))
print('Height/Width of each image: num_px = ' + str(num_px))
print('Each image is of size:(' + str(num_px) + ',' + str(num_px) + ',3)')
print('train_set_x shape:' + str(train_set_x_orig.shape))
print('train_set_y shape:' + str(train_set_y.shape))
print('test_set_x shape:' + str(test_set_x_orig.shape))
print('test_set_y shape:' + str(test_set_y.shape))
'''

#reshape the traning and test examples
#A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d,a) is to use:
#X_flatten = X.reshape(X.shape[0], -1).T
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

'''
print('train_set_x_flatten shape:' + str(train_set_x_flatten.shape))
print('train_set_y shape:' + str(train_set_y.shape))
print('test_set_x_flatten shape:' + str(test_set_x_flatten.shape))
print('test_set_y shape:' + str(test_set_y.shape))
print('sanity check after reshape:' + str(train_set_x_flatten[0:5,0]))
'''

#standardize the dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/(1 + np.exp(-z))
    return s
'''
print('sigmoid([0,2]) = ' + str(sigmoid(np.array([0,2]))))
'''

#GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim,1))
    b = 0
    
    assert w.shape == (dim,1)
    assert isinstance(b,float) or isinstance(b,int)
    
    return w,b
'''
dim = 2
w,b = initialize_with_zeros(dim)
print('w = ' + str(w))
print('b = ' + str(b))
'''

# GRADED FUNCTION: propagate
def propagation(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X) + b)   # compute activation
    cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)
    
    assert dw.shape == w.shape
    assert db.dtype == float
    
    cost = np.squeeze(cost)
    assert cost.shape == ()
    
    grads = {'dw':dw,
             'db':db}
    return grads, cost
''' 
w,b,X,Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagation(w,b,X,Y)
print('dw = ' + str(grads['dw']))
print('db = ' + str(grads['db']))
print('cost = ' + str(cost))
'''

# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagation(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads['dw']
        db = grads['db']
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print('Cost after iteration {}: {}'.format(i,cost))
    
    params = {'w':w,
              'b':b}
    
    grads = {'dw':dw,
             'db':db}
    
    return params, grads, costs

'''
params, grads, costs = optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False)
print('w = ' + str(params['w']))
print('b = ' + str(params['b']))
print('dw = ' + str(grads['dw']))
print('db = ' + str(grads['db']))
print(costs)
'''   

'''               
params, grads, costs = optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = True)
print('w = ' + str(params['w']))
print('b = ' + str(params['b']))
print('dw = ' + str(grads['dw']))
print('db = ' + str(grads['db']))        
'''
        
# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    assert Y_prediction.shape == (1,m)
    
    return Y_prediction
'''
print('prediction = ' + str(predict(w, b, X)))
'''

# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters['w']
    b = parameters['b']
    
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    # Print train/test Errors
    print('train accuracy: {} %'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {'costs': costs,
         'Y_prediction_test': Y_prediction_test,
         'Y_prediction_train': Y_prediction_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iterations': num_iterations}
    return d
'''
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,print_cost=True)
'''

'''  
# Plot learning curve (with costs)    
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()    
'''

'''
#Further analysis
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print('learning_rate is: ' + str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,
           test_set_y,num_iterations=1500,learning_rate=i,print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')
    
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]['costs']),label = str(models[str(i)]['learning_rate']))

plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc = 'upper center', shadow = True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()    
'''

'''
#Test with your own image
my_image = "timg-2.jpeg"   # change this to the name of your image file 

# preprocess the image to fit the algorithm
fname = 'images/' + my_image
image = np.array(plt.imread(fname))
my_image = transform.resize(image,(num_px,num_px)).reshape((1,num_px * num_px * 3)).T
my_predicted_image = predict(d['w'],d['b'],my_image)

plt.imshow(image)
print('y = ' + str(np.squeeze(my_predicted_image)) + 
      ", your algorithm predicts a \" " +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
'''        
    
    
    
    
    
    
    
            
    
    
    
    
    

    

