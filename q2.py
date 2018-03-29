# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.reshape(1,N_test)-y_test)**2).mean()          ##predictions are in (101,1,1) shape in three dimensions,
                                                                                ##but the y_test is in (101,1) shape, so I changed shape of predictions
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    from scipy.misc import logsumexp
    A = eye(x_train.shape[0])
    lse = logsumexp(-l2(test_datum.reshape(1,d),x_train)/2/tau**2)
    for i in range(x_train.shape[0]):
        A[i,i] = exp(-l2(test_datum.reshape(1,d),x_train[i,:].reshape(1,d))/2/tau**2 - lse)
    mat_x_train = mat(x_train)
    mat_A = mat(A)
    xTax = (mat_x_train.T)*mat_A*mat_x_train
    I = mat(np.eye(array(xTax).shape[0]))
    lamI = lam*I
    mat_y = mat(y_train[:,np.newaxis])
    xTay = (mat_x_train.T)*mat_A*mat_y
    w = np.linalg.solve((xTax+lamI),xTay) 
    y_hat = test_datum.T * w
    return y_hat
    ## TODO




def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    indice_tot = array(range(x.shape[0]))
    np.random.shuffle(indice_tot)
    new_losses = np.zeros(taus.shape)
    for i in range(k):
        indice_test_x = indice_tot[int(i*x.shape[0]/k):round((i+1)*x.shape[0]/k)]
        test_x = x[indice_test_x,:]
        test_y = y[indice_test_x]
        train_x = x[indice_tot[np.delete(indice_tot,array(range(int(i*x.shape[0]/k),round((i+1)*x.shape[0]/k))))]]
        train_y = y[indice_tot[np.delete(indice_tot,array(range(int(i*x.shape[0]/k),round((i+1)*x.shape[0]/k))))]]
        losses = run_on_fold(test_x, test_y, train_x, train_y, taus)
        print("l:",losses)    ##just check the running process
        new_losses = new_losses + losses
    return new_losses/k


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus,losses)
    plt.xlabel("taus")   ##add x,y label
    plt.ylabel("Losses")    ##add x,y label
    print("min loss = {}".format(losses.min()))

