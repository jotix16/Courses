import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output
#matplotlib inline

import warnings
warnings.filterwarnings("ignore")

#%load_ext autoreload
#%autoreload 2
from NN import *#

# Load dataset

X,Y=load_all_and_preproc()
Y = one_hot(10,Y)
X, Y, X_val, Y_val = create_val_set(X,Y)



# Test
X_test,Y_test,filenames_test = LoadBatch('test_batch')
Y_test = one_hot(10,Y_test)
X_test=preprocess(X_test)

print(X.shape,Y.shape, X_val.shape, Y_val.shape, X_test.shape,Y_test.shape)

# Parameters
n_in, n = X.shape
n_out = 10
n_hidden = 50

n_batch=100
n_s = 5*n/n_batch
n_epochs = int(2*n_s*n_batch/n)*5
print(n_epochs)

## Network
nn = Network()
nn.add(Dense(n_in, n_hidden)), nn.add(BatchNorm(n_hidden)), nn.add(ReLU())
nn.add(Dense(n_hidden, n_hidden)), nn.add(BatchNorm(n_hidden)), nn.add(ReLU())
nn.add(Dense(n_hidden, n_out))


#Train
nn.train(X, Y, X_val, Y_val, 
         shufle = True, n_epochs=n_epochs, retrain=False, eta=0.01, reg = 0.007653, visualize=False, n_batch=n_batch,
         eta_min=1e-5, eta_max=1e-1, n_s=n_s, augment = True, std = 0.12)


# Evaluate
nn.plot_training()
print("Train set accuracy:",nn.accuracy(X,Y))
print("Validation set accuracy:",nn.accuracy(X_val,Y_val))
print("Test set accuracy:",nn.accuracy(X_test,Y_test))
