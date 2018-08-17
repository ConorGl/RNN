# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:22:34 2018

@author: ConorGL

We will create the class RNNTheano, which will be an RNN
"""
import numpy as np
import Functions as fn

class RNNTheano:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign the instance variables 
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #Randomly initialize the network parameters, they depend upon the size
        #of the vocabulary set and the size (x,y) denotes the direction of 
        #forward propogation
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim, hidden_dim))
        
        def forward_propagation(self, x):
            # The total number of time steps
            T = len(x)
            # We start by creating a matrix of 0s in the shape of our input vector
            # + 1. This extra element will be used as the initial hidden layer
            # which will be set to 0 initially as the first input has no hidden
            # inputs yet.
            s = np.zeros((T + 1, self.hidden_dim))
            s[-1] = np.zeros(self.hidden_dim)
            # We now create the output as a 0 matrix in the shape of the input
            # vector by the number of words we have in the vocabulary
            o = np.zerios((T, self.word_dim))
            # Finally, we work out what we want to do for each time step
            # We are using tanh as our activation function 
            for t in np.arrange(T):
                s[t] = np.tanh(self.U[:,[t]] + self.W.dot(s[t-1]))
                o[t] = fn.softmax(self.V.dot(s[t]))
            # Both the hidden dim and current weights are output
            return [o, s]
        
        def predict(self, x):
            o, s = self.forward_propagation(x)
            return np.argmax(o, axis=1)
        
