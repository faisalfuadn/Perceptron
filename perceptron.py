# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:11:28 2020

@author: suantara
"""

"""
Simple Perceptron
Prove Perceptron only can solve linear problem
Input:
    -A : Int (from data)
    -B : Int (from data)
    -bias: Float (pre-determined)
Parameter:
    -learning_rate : how fast the model change (0-1)
    -Epochs= no of iteration (pre-determined)
    -no_input= total input number of data
"""
import numpy as np
class perceptron(object):
    def __init__(self, no_input, epochs=100, learning_rate=0.001):
        self.epochs= epochs
        self.learning_rate= learning_rate
        self.weight= np.random.normal(0, 0.5, size=(no_input + 1, ))
        self.error= []
        self.act_function='sigmoid'
        self.labels=[]
        
    def activation_function(self, value):
        """
        Parameters
        ----------
        value : value
            a value feeded to the selected activation function

        Returns
        -------
        value : int or float
            output value from the selected activation function

        """
        if self.act_function.lower() == 'unit_step':
            if value > 0:
                value= 1
            elif value < 0:
                value= 0
        elif self.act_function.lower() == 'linear':
            value= value
        
        elif self.act_function.lower() == 'sigmoid':
            value= 1/(1 +np.exp(-value))
        
        elif self.act_function.lower() == 'tanh':
            value= (np.exp(2*value)-1) / (np.exp(2*value)+1)
        
        elif self.act_function.lower() == 'relu':
            if value > 0:
                value= value
            elif value < 0:
                value= 0
        elif self.act_function.lower() == 'leaky_relu':
            if value >= 0:
                value= value
            elif value < 0:
                value= 0.1 * value
        elif self.act_function.lower() == 'swish':
            value= value * 1/(1 +np.exp(-value))
        elif self.act_function.lower() == 'mish':
            y= np.log(1+np.exp(value))
            value= value * ((np.exp(2*y)-1) / (np.exp(2*y)+1))
        else:
            print('act_function available: unit_step, linear, sigmoid, tanh /n')
        return value
    
    def learning(self, inputs):
        """

        Parameters
        ----------
        inputs : array of input

        Returns
        -------
        a float value

        """
        summation= np.dot(inputs, self.weight[1:]) + self.weight[0]
        activation= self.activation_function(value= summation)
        return activation
        
    def training (self, training_inputs, training_labels, name):
        """
        

        Parameters
        ----------
        training_inputs : np.array
            set of numpy array from the data
        training_labels : Int
            numerical categerocial type
        name : string
            activation function type

        Returns
        error: list
        TYPE
            list of error through iteration

        """
        self.act_function= name
        self.labels= list(set(training_labels))
        for  _ in range (self.epochs):
            err= 0
            for inputs, label in zip(training_inputs, training_labels):
                predict= self.learning(inputs)
                self.weight[0] += self.learning_rate * (label - predict)
                self.weight[1:] += self.learning_rate * (label - predict) * inputs
                err += err + abs(label-predict)
            self.error.append(err)
        return self.error
    
    def predict(self, inputs):
        """
        

        Parameters
        ----------
        inputs : list
            list of input

        Returns
        -------
        int
            label of input

        """
        predictions= self.learning(inputs)
        delta=[]
        for label in self.labels:
            delta.append(abs(label-predictions))
        idx= delta.index(min(delta))
        return self.labels[idx]
