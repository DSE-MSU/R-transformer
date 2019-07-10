from scipy.io import loadmat
import torch
import numpy as np


def data_generator(dataset, data_dir):
    print('loading Nott data...')
    data = loadmat(data_dir + 'Nottingham.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]
    
    for data in [X_train, X_valid, X_test]:
        print (dataset, len(data))
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test