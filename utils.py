#implemnt accuracy score
import numpy as np
import scipy.io as sio
 
def load_data_emo():
    '''
    load the data for emotion recognition.
    data has 200 subjects with 3 images each. 1st image is neutral, 2nd is emotion and 3rd is illumination
    return the X, y, X_test, y_test using only 1st and 2nd image as two classes
    '''
    pose = sio.loadmat('data/data.mat')
    faces = pose['face']
    faces = faces.reshape(-1, 600).T
    label = np.array([0, 1, 2]*200)

    # remove illumination
    ind = (label != 2)
    faces = faces[ind]
    label = label[ind]
    # split the data into train and test
    train_x = faces[:325]
    train_y = label[:325]
    test_x = faces[325:]
    test_y = label[325:]
    
    return train_x, train_y, test_x, test_y


def load_pose():
    '''
    pose has faces with 13 images of 68 subjects with first dimension as features
    return the X, y, X_test, y_test with 3 images of each subject as test data
    '''
    pose = sio.loadmat('data/pose.mat')
    faces = pose['pose']
    faces = faces.reshape(48*40, 13, 68)
    faces = np.concatenate(np.transpose(faces, (2, 1, 0)), axis=0)
    label = np.repeat(np.arange(68), 13)

    # split the data into train and test
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(68):
        train_x.append(faces[i*13:i*13+3])
        train_y.append(label[i*13:i*13+3])
        test_x.append(faces[i*13+3:i*13+13])
        test_y.append(label[i*13+3:i*13+13])
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    return train_x, train_y, test_x, test_y


    

def score(y, y_pred):
    '''
    Compute the accuracy score
    '''
    return (y==y_pred).sum()/len(y)

