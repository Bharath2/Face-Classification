import numpy as np
import matplotlib.pyplot as plt
from classifiers import bayes_classifier, knn_classifier
from feature_utils import PCA, MDA
from utils import *


#Use knn classifier. plot accuracy vs k on both train and test data
#Then use PCA and MDA to reduce the dimensionality of the data and plot the accuracy vs k on both train and test data
def PlotAccuracyVsK(x_train, y_train, x_test, y_test, label, DR, p_range):
    plt.figure(figsize=(10, 10))
    for p in p_range:
        x_train_p, y_train_p = x_train.copy(), y_train.copy()
        x_test_p, y_test_p = x_test.copy(), y_test.copy()

        dr = DR(x_train_p, y_train_p)
        x_train_p = dr.transform(x_train_p, p)
        x_test_p = dr.transform(x_test_p, p)

        train_acc = []
        test_acc = []
        for k in [1, 2, 5, 10, 20, 50]:
            model = knn_classifier(x_train_p, y_train_p, k)
            train_acc.append(model.accuracy(x_train_p, y_train_p))
            test_acc.append(model.accuracy(x_test_p, y_test_p))
        
        print('\n', label)
        print(p, [1, 2, 5, 10, 20, 50])
        print('Traning Accuracy \n', np.round(train_acc, 3))
        print('Testing Accuracy \n', np.round(test_acc, 3))

        plt.plot(['1', '2', '5', '10', '20', '50'], train_acc, label='Train data, dim = ' + str(p))
        plt.plot(['1', '2', '5', '10', '20', '50'], test_acc, label='Test data, dim = ' + str(p))

    plt.legend()
    plt.xlabel('K nearest neighbors')
    plt.ylabel('Accuracy')
    if label != '': label = ' ' + label + ' + '
    plt.title(label + 'KNN + Emotion classification')
    plt.savefig(label + 'knn.png')
    plt.show()

#use PCA to reduce the dimensionality of the data and plot the accuracy vs k on both train and test data
#load data
x_train, y_train, x_test, y_test = load_pose()
PlotAccuracyVsK(x_train, y_train, x_test, y_test, 'PCA', PCA, [1, 2, 10, 50, 100])
PlotAccuracyVsK(x_train, y_train, x_test, y_test, 'MDA', MDA, [1, 2, 10, 50, 100])
 
    
# x_train, y_train, x_test, y_test = load_data_emo()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# PlotAccuracyVsK(x_train, y_train, x_test, y_test, 'PCA', PCA, [1, 2, 10, 50, 100])
# PlotAccuracyVsK(x_train, y_train, x_test, y_test, 'MDA', MDA, [1, 2, 10, 50, 100])
 