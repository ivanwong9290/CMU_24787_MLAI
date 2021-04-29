#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:13:33 2021

@author: lbk
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 50)
    y = np.linspace(ylim[0], ylim[1], 50)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    I1 = np.where(P> 1)
    I1 = np.reshape(I1,(-1,1))
    I2 = np.where(P<-1)
    I2 = np.reshape(I2,(-1,1))
    
    #ax.scatter(X[I1], Y[I1], s=1, marker='.', color = 'red' , alpha=0.1)
    #ax.scatter(X[I2], Y[I2], s=1, marker='.', color = 'green' , alpha=0.1)
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, color = 'blue', facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()