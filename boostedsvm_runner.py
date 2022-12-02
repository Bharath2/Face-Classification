import numpy as np
from matplotlib import pyplot as plt
from feature_utils import PCA, MDA
from classifiers import Boosted_svm_classifier
from utils import *

def PlotAccuracyVsK(x_train, y_train, x_test, y_test, label, DR, p_range):
    train_acc = []
    test_acc = []
    dr = DR(x_train, y_train.copy())
    for p in p_range:

        x_train_p = dr.transform(x_train, p)
        x_test_p = dr.transform(x_test, p)

        model = Boosted_svm_classifier(x_train_p, 2*y_train-1)
        model.train()
        train_acc.append(model.accuracy(x_train_p, 2*y_train-1))
        test_acc.append(model.accuracy(x_test_p, 2*y_test - 1))
    
    print('\n', label)
    print(p_range)
    print('Traning Accuracy \n', np.round(train_acc, 3))
    print('Testing Accuracy \n', np.round(test_acc, 3))
        
    plt.plot(p_range, train_acc, label='Train data')
    plt.plot(p_range, test_acc, label='Test data')
    plt.legend()
    plt.xlabel('Dimensionality')
    plt.ylabel('Accuracy')
    if label != '': label = ' ' + label + ' + '
    plt.title(label + 'Boosted SVM + Emotion classification')
    plt.savefig(label + 'bsvm.png')
    plt.show()

x_train, y_train, x_test, y_test = load_data_emo()
PlotAccuracyVsK(x_train, y_train, x_test, y_test, 'PCA', PCA, [1, 2, 10, 50, 100])
PlotAccuracyVsK(x_train, y_train, x_test, y_test, 'MDA', MDA, [1, 2, 10, 50, 100])