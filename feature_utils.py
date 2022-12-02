#implement PCA and MDA

import numpy as np
import matplotlib.pyplot as plt

class PCA():
    '''
    Perform PCA on the data x
    '''
    def __init__(self, x, y):
        self.x = x
        self.x = self.x - self.x.mean(axis=0)
        self.cov = np.cov(self.x, rowvar=False)
        self.evals, self.evecs = np.linalg.eigh(self.cov)
        self.evals = self.evals[::-1]
        self.evecs = self.evecs[:,::-1]
 
    def transform(self, x, k):
        '''
        Transform the data x using the first k eigenvectors
        '''
        return np.matmul(x, self.evecs[:,:k])

class MDA():
    '''
    Perform Maximal Discriminant Analysis on the data x
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x = self.x - self.x.mean(axis=0)
        self.cov = np.cov(self.x, rowvar=False)
        # Compute the mean of each class
        self.classes = np.unique(self.y)
        self.means = np.zeros((len(self.classes), self.x.shape[1]))
        for i, c in enumerate(self.classes):
            self.means[i] = self.x[self.y == c].mean(axis=0)
        # Compute the between class scatter matrix
        self.Sb = np.zeros((self.x.shape[1], self.x.shape[1]))
        for i, c in enumerate(self.classes):
            self.Sb += len(self.x[self.y == c]) * np.outer(self.means[i] - self.x.mean(axis=0), self.means[i] - self.x.mean(axis=0))
        # Compute the within class scatter matrix
        self.Sw = np.zeros((self.x.shape[1], self.x.shape[1]))
        for i, c in enumerate(self.classes):
            self.Sw += np.cov(self.x[self.y == c], rowvar=False)
        # Compute the eigenvalues and eigenvectors of the generalized eigenvalue problem
        self.evals, self.evecs = np.linalg.eigh(np.matmul(np.linalg.pinv(self.Sw), self.Sb))
        self.evals = self.evals[::-1]
        self.evecs = self.evecs[:,::-1]

    def transform(self, x, k):
        '''
        Transform the data x using the first k eigenvectors
        '''
        return np.matmul(x, self.evecs[:,:k])
