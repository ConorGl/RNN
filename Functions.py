# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:11:47 2018

@author: ConorGL

We want to create some extra functions to help our model.
"""
import numpy as np

#The soft max function will be used to 'smooth' our outputs so that a single
#word is chosen after each input
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

