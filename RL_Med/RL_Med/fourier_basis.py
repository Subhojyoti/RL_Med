import numpy as np
import sys
import random


import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

if "../" not in sys.path:
    sys.path.append("../") 

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS

from scipy.fftpack import fft, dct




class FourierSampler():
    
    def __init__(self, constant=1, n_components=100, random_state=None):
        
        #S = np.zeros((n_components, n_components))
        self.n_components = n_components
        self.constant = constant
        self.random_state = random_state
        
        
    
    def fit(self, X, y=None, n_order = 5):
        """Fit the model with X.
        Samples random projection according to n_features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """

        X = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]
        
        #self.freqs = np.fft.rfftfreq(100, d=1.0)
        '''
        self.C = np.zeros((n_features, self.n_components))
        
        ns = np.arange(self.n_components)
        
        one_cycle = dct(ns, 1)
        print(one_cycle)
        #for ord in range(1,n_order+1):
        #    one_cycle += ord * self.constant * np.pi * ns / self.n_components
            
        for k in range(0, n_features):
            t_k = k * one_cycle
            self.C[k, :] = np.cos(t_k)
        '''
        
        self.C = np.zeros((n_features, self.n_components))
        ns = np.arange(self.n_components)
        
        print(ns)
        one_cycle = 2 * self.constant * np.pi * ns / self.n_components    
        '''
        for k in range(1,int(self.constant)):
            one_cycle += 2 * np.pi * ns / self.n_components  
            
        print(one_cycle)
        '''
        
        #one_cycle = 2 * self.constant * np.pi * ns / self.n_components   
        
        for k in range(0, n_features):
            t_k = k * one_cycle
            self.C[k, :] = np.cos(t_k)
        
        self.random_weights_ = self.C
        
        #self.random_weights_ = self.C
        #
        #self.random_weights_ = (dct(random_state.normal(size=(n_features, self.n_components))))
        '''
        self.random_weights_ = [np.fft.rfftfreq(100).tolist()[0] for m in range(0,n_features)]
        
        print(self.random_weights_)
        print(self.freqs)
        '''
        self.random_offset_ = random_state.uniform(0, 2 * np.pi,size=self.n_components)
        #self.random_offset_ = random_state.uniform(0, 0)
        
        return self
    
    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self, 'random_weights_')

        X = check_array(X, accept_sparse='csr')
        projection = safe_sparse_dot(X, self.random_weights_)
        projection += self.random_offset_
        np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection
    
    '''
    def transform(self, X):
        nsamples, nfeatures = X.shape
        nfreqs = len(self.random_weights_)
        """Given a list of original data, return a list of feature vectors."""
        features1 = np.sin(2. * np.pi * self.freqs[None, None, :] * X[:, :,None]).reshape(nsamples, nfeatures * nfreqs)
        features2 = np.cos(2. * np.pi * self.freqs[None, None, 1:] * X[:, :,None]).reshape(nsamples, nfeatures * (nfreqs-1))
        features = np.concatenate([features1, features2], axis=1)
        print(features)
        return features
    '''