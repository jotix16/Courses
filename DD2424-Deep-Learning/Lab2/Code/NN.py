import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output


class Layer:
    #A building block. Each layer is capable of performing two things:
    #- Process input to get output:           output = layer.forward(input)
    #- Propagate gradients through itself:    grad_input = layer.backward(input, grad_output)
    #Some layers also have learnable parameters which they update during layer.backward.
    
    def __init__(self):
        # Here we can initialize layer parameters (if any) and auxiliary stuff.
        # A dummy layer does nothing
        pass
    
    def forward(self, input):
        # Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        # A dummy layer just returns whatever it gets as input.
        return input

    def backward(self, input, grad_output):
        # Performs a backpropagation step through the layer, with respect to the given input.
        # To compute loss gradients w.r.t input, we need to apply chain rule (backprop):
        # d loss / d x  = (d loss / d layer) * (d layer / d x)
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input) # chain rule


class ReLU(Layer):
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        self.fc = False # whether it is fully connected layer
    
    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
#         return np.dot(grad_output,relu_grad)
        return grad_output*relu_grad



class Dense(Layer):
    """ Standard fully connected layer."""
    def __init__(self, input_units, output_units ):
        self.fc = True # whether it is fully connected layer
        self.ins = input_units
        self.outs = output_units
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(1/input_units), # Suggested  init
                                        size = (output_units, input_units))
#                        np.random.normal(loc=0.0, 
#                                         scale = np.sqrt(2/(input_units+output_units)), # He init
#                                         size = (output_units, input_units))
        self.biases = np.zeros(output_units)
#         np.random.normal(loc=0.0, 
#                                         scale = np.sqrt(2/(output_units)), # He init
#                                         size = output_units)
    
    def forward(self,input):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        # input shape: [input_units, batch]
        # output shape: [output units, batch]
        
        if input.ndim == 1:
            input = input.reshape(input.shape[0],1)
            
        return np.dot(self.weights, input) + self.biases[:,np.newaxis]
    
    def backward(self,input,grad_output,eta=0.001, regularization = 0.1):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        if input.ndim == 1:
            input = input.reshape(input.shape[0],1)
        grad_input = np.dot(self.weights.T, grad_output)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(grad_output, input.T)/(input.shape[1]) 
        if regularization:
            grad_weights += 2 * regularization * self.weights
            
        grad_biases = grad_output.mean(axis=1)
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - eta * grad_weights
        self.biases = self.biases - eta * grad_biases
            
        return grad_input 



class Network():
    def __init__(self,layers = []):
        self.layers = layers[:]
        self.startup()
        
    def add(self,layer):
        self.layers.append(layer)
    
    def startup(self,n_epochs=40, n_batch=100, eta=0.001, reg=0.01, eta_min=None, eta_max=None, n_s=None):
        # Main param
        self.n_batch = n_batch
        self.eta = eta # learning rate
        self.reg = reg # regularization of weights
        
        ## Cyclic learning rate
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_s = n_s
        self.t = 0 # nr of updates
        
        ## Logging
        self.train_acc_log = []
        self.val_acc_log = []
        self.train_loss_log = []
        self.val_loss_log = []
        self.train_cost_log = []
        self.val_cost_log = []
        self.eta_log = []
        self.t = 0
        
    def re_init(self):
        for l in self.layers:
            if l.fc:
                l.__init__(l.ins,l.outs)
                
    #############################################    
    ######## Cross Entropy loss function ########    
    def CrossEntropyLoss(self, softmax_input, P, Y_batch):
        lossgrad = -(Y_batch - P)
        tmp = np.exp(softmax_input)
        return lossgrad
    
    def SoftMax(self, input):
        tmp = np.exp(input)
        softmax = tmp / np.sum(tmp,0)[np.newaxis,:]
        return softmax
       
    def loss(self,X_batch, Y_batch):              
        activations = self.forward(X_batch)
        softmax_input = activations[-1]
        tmp = np.exp(softmax_input)
        loss_matrix = - softmax_input + np.log(np.sum(tmp,0))[np.newaxis,:]
        return (loss_matrix * Y_batch).sum()/Y_batch.shape[1]
    #############################################
    
    
    #############################################    
    ########           Logging           ########                  
    def logging(self, X_train, Y_train_hot, X_valid, Y_valid_hot, visualize, epoch):
        ll = 0
        for l in self.layers:
            if l.fc:
                ll +=self.reg*np.sum(l.weights[:]**2)
        self.train_acc_log.append(self.accuracy(X_train, Y_train_hot))
        self.val_acc_log.append(self.accuracy(X_valid, Y_valid_hot))
        self.train_loss_log.append(self.loss(X_train,Y_train_hot))
        self.val_loss_log.append(self.loss(X_valid,Y_valid_hot))
        self.train_cost_log.append(self.train_loss_log[-1] + ll)
        self.val_cost_log.append(self.val_loss_log[-1] + ll)
        if visualize:
            clear_output(wait="True")
        print("Epoch",epoch)
        print("Train accuracy:", self.train_acc_log[-1],"Train loss:","%.4f" % self.train_loss_log[-1] )
        print("Val accuracy:", self.val_acc_log[-1], "Val loss:", "%.4f" % self.val_loss_log[-1],"\n")
        if visualize:
            self.plot_training()
                
    def plot_training(self):
        fig, axs = plt.subplots(1,3, figsize=(15, 4))
        fig.subplots_adjust(hspace = .5, wspace=0.3)
        axs = axs.ravel()


        axs[0].plot(self.train_cost_log,label='train cost')
        axs[0].plot(self.val_cost_log,label='val cost')
        axs[0].legend(loc='best')
        axs[0].grid()
        
        axs[1].plot(self.train_loss_log,label='train loss')
        axs[1].plot(self.val_loss_log,label='val loss')
        axs[1].legend(loc='best')
        axs[1].grid()

        axs[2].plot(self.train_acc_log,label='train accuracy')
        axs[2].plot(self.val_acc_log,label='val accuracy')
        axs[2].legend(loc='best')
        axs[2].grid()
        plt.show()
    #############################################
    
    def forward(self, input):
        tmp = input
        activations = []
        activations.append(tmp)
        for l in self.layers:
            tmp = l.forward(activations[-1])
            activations.append(tmp)
        return activations
    
    def backward(self, X_batch, Y_batch):
        if Y_batch.ndim == 1:
            Y_batch = Y_batch.reshape(Y_batch.shape[0],1)
        activations = self.forward(X_batch)
        lossgrad = self.CrossEntropyLoss(activations[-1], self.SoftMax(activations[-1]), Y_batch)
        
        for i, l in reversed(list(enumerate(self.layers))):
            if l.fc:
                lossgrad = l.backward(activations[i], lossgrad, self.eta, self.reg)
            else:
                lossgrad = l.backward(activations[i],lossgrad)
    
    def predict(self, input):
        p = self.SoftMax(self.forward(input)[-1])
        return p.argmax(0)
    
    def accuracy(self, X, Y):
        Y_pred = self.predict(X)
        if Y.ndim == 2:
            Y = Y.argmax(0)
        return np.mean(Y_pred == Y)
    
    def cyclical_learning_rate(self):
        if self.eta_min and self.eta_max and self.n_s:
            t = np.mod(self.t, 2*self.n_s)
            self.eta = self.eta_min + np.copysign(np.mod(t,self.n_s),self.n_s-t)*(self.eta_max- self.eta_min)/self.n_s + (self.n_s<=t)*(self.eta_max-self.eta_min)
            self.eta_log.append(self.eta)

    def minibatch_SGD(self, X_train, Y_train_hot):
        n = X_train.shape[1]
        for j in range(0,n, self.n_batch):
                    self.cyclical_learning_rate() # updates the learning rate
                    X_batch = X_train[:, j:j+self.n_batch];
                    Y_batch = Y_train_hot[:, j:j+self.n_batch];
                    loss = self.backward(X_batch, Y_batch)
                    #self.t = np.mod(self.t+1, 2*self.n_s) # increase update nr
                    self.t += 1

    def augment_batch(self, X_batch,std = 0.1):
        return X_batch + np.random.normal(scale = std, size=X_batch.shape) # variance = std^2
    
    def train(self, X_train, Y_train_hot, X_valid, Y_valid_hot, 
              shufle=True, n_epochs=40, n_batch=100, eta=0.001, reg=0.01, visualize=False, 
              eta_min=None, eta_max=None, n_s=None, augment = False, ensamble = True, ensamble_list=None, ensamble_name ="ensamble"):
        
        # Delete old logging
        self.startup(n_epochs, n_batch, eta, reg, eta_min, eta_max, n_s)
        
        # Reiinit weighs
        self.re_init()
        
        for epoch in tqdm(range(n_epochs)):
            if shufle:
                X_train, Y_train_hot = shuffle(X_train,Y_train_hot) # shuffle

            ## Minibatch SGD
            self.minibatch_SGD(X_train, Y_train_hot)
            
            ## Logging
            self.logging(X_train, Y_train_hot, X_valid, Y_valid_hot, visualize,epoch+1)
            
            ## Ensamble
            if ensamble_list is not None:
                if epoch+1 in ensamble_list:
                    print("Save NN for ensamble in epoch:",epoch+1)
                    pickle.dump(self, open("Networks/"+ensamble_name+str(epoch+1)+".p", "wb"))
                    
                    
                    
# ===========================================================================#
def ensamble_prediction(nns,XX,YY):
    prediction_list = [nn.predict(XX) for nn in nns]
    tt = one_hot(10,prediction_list).transpose(1,2,0)
    aa = np.sum(tt,axis=0)
    ans = np.argmax(aa,1)
    if YY.ndim == 2:
        YY = list(YY.argmax(0))
        acc = np.mean(ans == YY[:])
    return acc

def shuffle(X,Y):
    indexes = np.random.permutation(X.shape[1])
    X = np.take(X,indexes,axis=1)
    Y = np.take(Y,indexes,axis=1)
    return X,Y

def create_val_set(X,Y,n_val=5000):
    #X, Y = shuffle(X,Y)
    X_val = X[:,0:n_val]
    Y_val = Y[:,0:n_val]
    X = X[:,n_val:]
    Y = Y[:,n_val:]
    return X, Y, X_val, Y_val