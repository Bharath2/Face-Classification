#implement bayes and knn classifiers

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, mode

class bayes_classifier():
        
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.classes = np.unique(y)
            
        def train(self):
            '''
            Train the classifier by computing the mean and covariance of each class
            '''
            self.means = []
            self.covs = []
            self.priors = []
            for c in self.classes:
                x_c = self.x[self.y==c]
                self.means.append(x_c.mean(axis=0))
                self.covs.append(np.cov(x_c.T) + 1e-5*np.eye(x_c.shape[1]))
                self.priors.append(len(x_c)/len(self.x))
        
        def predict(self, x_test): 
            '''
            Predict the class of each sample in x_test
            '''
            discr = []
            for c in range(len(self.classes)):
                discr.append(multivariate_normal.logpdf(x_test, self.means[c], self.covs[c], allow_singular=1) + np.log(self.priors[c]))
            return self.classes[np.argmax(discr, axis=0)]
        
        def accuracy(self, x_test, y_test):
            '''
            Compute the accuracy of the classifier
            '''
            return np.mean(self.predict(x_test) == y_test)


class knn_classifier():
        
        def __init__(self, x, y, k = 3):
            self.x = x
            self.y = y
            self.k = k
            self.classes = np.unique(y)
            
        def predict(self, x_test):
            '''
            Predict the class of each sample in x_test
            '''
            #use cdist to compute the distance between each pair of points
            dist = np.sqrt(np.sum((x_test[:,None,:] - self.x[None,:,:])**2, axis=2))
            #find the k nearest neighbors
            idx = np.argsort(dist, axis=1)[:, :self.k]
            #find the most common class among the k nearest neighbors using mode function
            return mode(self.y[idx], axis=1)[0].flatten()
        
        def accuracy(self, x_test, y_test):
            '''
            Compute the accuracy of the classifier
            '''
            return np.mean(self.predict(x_test) == y_test)


#implement kernel svm classifier
class kernel_svm_classifier():
        
        def __init__(self, x, y, kernel = 'rbf', C = 1, sigma = 1, r = 1):
            self.x = x
            self.y = y
            self.kernel = kernel
            self.C = C
            self.sigma = sigma
            self.r = r
            self.classes = np.unique(y)
        
        def Kernel(self, x1, x2):
            '''
            Compute the kernel between x1 and x2
            '''
            if self.kernel == 'rbf':
                return np.exp(-np.sum((x1 - x2)**2)/(self.sigma**2))
            elif self.kernel == 'poly':
                return (np.dot(x1, x2.T) + 1)**self.r
        
        def train(self):
            '''
            using dual optimization with minimize function
            '''
            #define the objective function
            def objective(alpha):
                return 0.5*np.sum(alpha**2) - np.sum(alpha)
            #define the constraint
            def constraint(alpha):
                return np.sum(alpha*self.y)
            #define the bounds
            bounds = [(0, self.C) for i in range(len(self.x))]
            #define the initial guess
            alpha0 = np.zeros(len(self.x))
            #define the constraint
            cons = {'type': 'eq', 'fun': constraint}
            #optimize the objective function
            res = minimize(objective, alpha0, bounds = bounds, constraints = cons)
            #get the optimal alpha
            self.alpha = res.x
            #compute the bias
            self.b = np.mean(self.y - np.dot(self.alpha*self.y, self.Kernel(self.x, self.x)))
        
        def cross_validation_train(self, k = 3):
            '''
            Choose the best hyperparameters using cross validation
            '''
            #split the data into k folds
            x_folds = np.array_split(self.x, k)
            y_folds = np.array_split(self.y, k)
            #define the range of hyperparameters
            if self.kernel == 'rbf':
                param_range = [0.01, 1, 2, 3, 5, 7, 10]
            elif self.kernel == 'poly':
                param_range = np.arange(1, 10)
            
            #define the best hyperparameters
            best_param = 0
            best_acc = 0
            #loop over the hyperparameters
            for param in param_range:
                #set the hyperparameters
                if self.kernel == 'rbf':
                    self.sigma = param
                elif self.kernel == 'poly':
                    self.r = param
                #initialize the accuracy
                acc = 0
                #loop over the folds
                for i in range(k):
                    #split the data into training and validation sets
                    self.x = np.concatenate(x_folds[:i] + x_folds[i+1:])
                    self.y = np.concatenate(y_folds[:i] + y_folds[i+1:])
                    x_val = x_folds[i]
                    y_val = y_folds[i]
                    #train the classifier
                    self.train()
                    #compute the accuracy
                    acc += self.accuracy(x_val, y_val)
                #update the best hyperparameters
                if acc > best_acc:
                    best_param = param
                    best_acc = acc
            #set the best hyperparameters
            print('best param:', best_param)
            if self.kernel == 'rbf':
                self.sigma = best_param
            elif self.kernel == 'poly':
                self.r = best_param
            #train the classifier with the best hyperparameters
            self.x = np.concatenate(x_folds)
            self.y = np.concatenate(y_folds)
            self.train()
            

        def predict(self, x_test):
            '''
            Predict the class of each sample in x_test
            '''
            #compute the kernel between x_test and x
            K = np.zeros((len(x_test), len(self.x)))
            for i in range(len(x_test)):
                for j in range(len(self.x)):
                    K[i,j] = self.Kernel(x_test[i], self.x[j])
            #compute the prediction
            return np.sign(np.dot(K, self.alpha*self.y) + self.b)

        def accuracy(self, x_test, y_test):
            '''
            Compute the accuracy of the classifier
            '''
            return np.mean(self.predict(x_test) == y_test)

class linear_svm():

    def __init__(self, x, y, c = 1, w = None, b = None):
        self.x = x
        self.y = y
        self.c = c
        self.w = w
        self.b = b
        self.classes = np.unique(y)
    
    def train(self, weights = None):
        '''
        train linear svm with weights given to each data point for cost function
        '''
        #define the cost function with weights from adaboost
        if weights is None:
            weights = np.ones(len(self.x))
        def cost(w):
            return np.sum(weights*(np.maximum(1 - self.y*(np.dot(self.x, w)), 0)**2)) + self.c*np.sum(w**2)
        #optimize the cost function
        res = minimize(cost, np.zeros(len(self.x[0])))
        #get the optimal w and b
        self.w = res.x
        self.b = np.mean(self.y - np.dot(self.x, self.w))

    
    def predict(self, x_test):
        '''
        Predict the class of each sample in x_test
        '''
        return np.sign(np.dot(x_test, self.w) + self.b)
    
    def accuracy(self, x_test, y_test):
        '''
        Compute the accuracy of the classifier
        '''
        return np.mean(self.predict(x_test) == y_test)

#implement boosted svm classifier with linear svms
class Boosted_svm_classifier():
            
            def __init__(self, x, y, C = 1, T = 10):
                self.x = x
                self.y = y
                self.C = C
                self.T = T
                self.classes = np.unique(y)
            
            def train(self):
                '''
                Train the classifier
                '''
                #initialize the weights
                self.w = np.ones(len(self.x))/len(self.x)
                self.alphas = []
                self.svms = []
                for t in range(self.T):
                    #train a linear svm with the current weights
                    svm = linear_svm(self.x, self.y, c = self.C)
                    svm.train(self.w)
                    #compute the error
                    y_pred = svm.predict(self.x)
                    error = np.sum(self.w[y_pred != self.y])
                    #compute the alpha
                    alpha = 0.5*np.log((1-error)/error)
                    self.alphas.append(alpha)
                    self.svms.append(svm)
                    #update the weights
                    self.w *= np.exp(-alpha*self.y*y_pred)
                    self.w /= np.sum(self.w)
            
            def predict(self, x_test):
                '''
                Predict the class of each sample in x_test
                '''
                #compute the prediction
                y_pred = np.zeros(len(x_test))
                for t in range(self.T):
                    y_pred += self.alphas[t]*self.svms[t].predict(x_test)
                return np.sign(y_pred)
    
            def accuracy(self, x_test, y_test):
                '''
                Compute the accuracy of the classifier
                '''
                return np.mean(self.predict(x_test) == y_test)
