import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset

def GenerateData():

    ### statistical repartition of groups points
    mu_A = np.array([[3],[3]])
    sigma_A = 0.5

    mu_B = np.array([[0],[2]])
    sigma_B = 0.3

    mu_C = np.array([ [2] ,[0] ])
    sigma_C = 0.7


    # La convention Pytorch exige qu'on ait un élément par ligne => pour dire qu'on ne doit pas structurer les état d'un élément par colonne, mais je ne le fais pas, j'inverserai plus tard
    ### generation of 50 points of each of the 3 groups
    X_A = sigma_A*np.random.randn(2,50) + mu_A
    X_B = sigma_B*np.random.randn(2,50) + mu_B
    X_C = sigma_C*np.random.randn(2,50) + mu_C

    Y_A = 0 * np.ones( (X_A.shape[1],1) ).squeeze()
    Y_B = 1 * np.ones( (X_B.shape[1],1) ).squeeze()
    Y_C = 2 * np.ones( (X_C.shape[1],1) ).squeeze()


    
    ### visualisating the data 
    fig, ax = plt.subplots()
    plt.scatter(X_A[0,:],X_A[1,:], label = "group A")
    plt.scatter(X_B[0,:],X_B[1,:], label = "group B")
    plt.scatter(X_C[0,:],X_C[1,:], label = "group C")
    plt.legend()
    ax.set_xlabel('x_1'), ax.set_ylabel('x_2')
    #ax.set_xlim((-5,10)), ax.set_ylim((0,10))
    fig.suptitle('created dataset')
    plt.show()


    ### creation of the training dataset via concatenation and conversion in Pytorch format
    X_train_np = np.hstack( (X_A,X_B,X_C) )
    Y_train_np = np.concatenate( (Y_A,Y_B,Y_C) )
    #Y_train_np = np.resize( Y_train_np, (1,Y_train_np.shape[0]) )


    X_train = torch.from_numpy( X_train_np )
    Y_train = torch.from_numpy( Y_train_np ).flatten()
    print(X_train.shape), print( Y_train.shape )

    # The TensorDataset constructor imposes that the 2 tensors must share the same 1st dim so we transpose the 3 matrices
    X_train = X_train.T
    Y_train = Y_train.T
    train_ds = TensorDataset(X_train, Y_train)          # the TensorDatset structure allows to associate a point with its corresponding output 
   
    ### test with the TensorDatset structure
    # bs = 3
    # xb,yb = train_ds[0:bs]
    # print(f'\n the first 3 points of the dataset : \n {xb}')
    # print(f'\n the associated outputs of those points 3 : \n {yb}')

    return train_ds

if __name__ == "__main__":
    GenerateData()