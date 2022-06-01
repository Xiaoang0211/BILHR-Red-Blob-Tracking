#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:30:22 2022

@author: xiaoang zhang, leon mayer
"""

import numpy as np
import math

class CMAC:

    def __init__(self, n_input, n_output, n_a, res, disp, maxi, mini):
        self.n_y = n_input #2
        self.n_x = n_output #2
        self.n_a = n_a #3
        self.res = res # the quantization of input, 50 for both res = [50 , 50]
        self.disp = disp # displacement

        n_w = self.n_a
        for i in range(0, self.n_y):
            n_w = n_w * math.floor(((res[i] - 2)/self.n_a) + 2)
        self.n_w = int(n_w)
        self.alpha = 0.3 # learning rate     
        self.mu = [0] * self.n_a
        self.mini = mini
        self.maxi = maxi
        self.W = np.zeros([self.n_x, self.n_w])
        
        # map of the input output data
    def cmacMap(self, y):
        mu = [int(a) for a in self.cmacQuantizeAndAssociate(y)]
        x = [0] * self.n_x
        for i in range(0, self.n_x): # n_x = 2
            for j in range(0, self.n_a):
                x[i] = x[i] + self.W[i, mu[j]]
        return x
        
        #compute the weights this is the equivalent to L2 of neural network

    def cmacQuantizeAndAssociate(self, y_sample): # y = [y1, y2], y1 = [y1,1 ... y1,75]
        q = np.zeros([self.n_y, 1])
        p = np.zeros([self.n_a, self.n_y])
        for i in range(0, self.n_y): # n_y = 2
            if y_sample[i] < self.mini[i]:
                y_sample[i] = self.mini[i]
            if y_sample[i] > self.maxi[i]:
                y_sample[i] = self.maxi[i]
            # quantization of input
            q[i] = math.floor(self.res[i]*((y_sample[i] - self.mini[i])/(self.maxi[i] - self.mini[i])))
        
            if q[i] >= self.res[i]:
                q[i] = self.res[i] - 1
            
        for i in range(0, self.n_a): # n_a = 3
            for j in range(0, self.n_y): # n_y = 2
                p[i, j] = math.floor((q[j]+self.disp[i, j])/self.n_a)
            p_i = p[i, :]
            h = self.cmacHash(p_i) # weight index for AU i
            self.mu[i] = int(h)
        return self.mu

        # compute a virtual address for weight table indexes p[i, 1]...p[i, n_y] 
    def cmacHash(self, p_i):
        h = 0
        r = [0] * self.n_y
        for i in range(0, self.n_y):
            r[i] = math.floor((self.res[i]-2)/self.n_a) + 2
        # calculating the hash address
        for j in range(0, self.n_y):
            h = h*r[j] + p_i[j]
        return h
        
    
    # give the target and learning rate, updating the weights
    def cmacTargetTrain(self, targets, x):
        for i in range(0, self.n_x):
            inc = self.alpha*((targets[i]-x[i])/self.n_a)
            for j in range(0, self.n_a):
                self.W[i, self.mu[j]] = self.W[i, self.mu[j]] + inc
        return self.W
