import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm
from IPython.display import clear_output

############################################
########           Layers           ########  
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
        self.fc = 'RL' # whether it is fully connected layer

    def forward(self, input):
        # Apply elementwise ReLU to [input_units, batch] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(Layer):
    """ Standard fully connected layer."""
    def __init__(self, input_units, output_units, init ='He',std=None ):
        self.fc = 'FC' # whether it is fully connected layer
        self.ins = input_units
        self.outs = output_units
        if std:
            print("init gauss:", std)
            self.weights = np.random.normal(loc=0.0, 
                                            scale = np.sqrt(std),
                                            size = (output_units, input_units))
        elif init == 'Xavier':
            self.weights = np.random.normal(loc=0.0, 
                                            scale = np.sqrt(1/input_units),
                                            size = (output_units, input_units))
        else:# init == 'He':
            self.weights = np.random.normal(loc=0.0, 
                                            scale = np.sqrt(2/input_units),
                                            size = (output_units, input_units))
        self.biases = np.zeros(output_units)

    def forward(self,input):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        # input shape: [input_units, batch]
        # output shape: [output units, batch]
        if input.ndim == 1:
            input = input.reshape(input.shape[0],1)
        return np.dot(self.weights, input) + self.biases[:,np.newaxis]
    
    def backward(self,input,grad_output,eta=0.001, regularization = 0.1):        
        if input.ndim == 1:
            input = input.reshape(input.shape[0],1)
        grad_input = np.dot(self.weights.T, grad_output)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(grad_output, input.T)/(input.shape[1]) 
        if regularization:
            grad_weights += 2 * regularization * self.weights
        grad_biases = grad_output.mean(axis=1) # mean over batch axis
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - eta * grad_weights
        self.biases = self.biases - eta * grad_biases
        return grad_input 

class BatchNorm(Layer):
    """ Standard fully connected layer."""
    def __init__(self, out_units, alpha=0.9):
        self.fc = 'BN' # whether it is fully connected layer
        self.eps = np.finfo(np.float).eps
    
        # Params of Batch Norm layer
        self.gamma = np.ones(out_units) # scale, equivalent to->std(var^0.5)
        self.beta = np.zeros(out_units) # shift
        self.alpha = alpha 
        
        # Needed for Backprop during training
        self._s_hat = None
        self. first_update=True
        self._mu_b = None
        self._var_b_2 = None
        
        # Needed during testing
        self._mu_av = np.zeros(out_units) 
        self._var_av = np.zeros(out_units) 
         
    def forward(self,X,training = False):
        # Perform Batch Normalization followed by shift and scale
        # X shape: [output_units, batch] which is the output of the FC layer before
        if X.ndim == 1: X = X.reshape(X.shape[0],1)
        if training: 
            # Batch normalization
            self._mu_b = X.mean(1, dtype=np.float64) # needed in backprop
            self._var_b_2 = ((X - self._mu_b[:,np.newaxis])**2).mean(1, dtype=np.float64) # var
            if self.first_update: # first minibatch update
                self._mu_av = self._mu_b
                self._var_av = self._var_b_2
            else: # exponential moving
                self._mu_av = self.alpha*self._mu_av  + (1-self.alpha)*self._mu_b
                self._var_av = self.alpha*self._var_av  + (1-self.alpha)*self._var_b_2 
            X = (X - self._mu_b[:,np.newaxis]) * (self._var_b_2[:, np.newaxis] +self.eps)**-0.5
            self._s_hat = np.array(X) # update _s_hat needed for backprop
        else:
            X = (X - self._mu_av[:,np.newaxis]) * (self._var_av[:, np.newaxis] +self.eps)**-0.5
        # Shift and scale
        X = X * self.gamma[:, np.newaxis] + self.beta[:,np.newaxis]
        return X
    
    def backward(self, X, grad_output, eta):
        ## Shift and scale
        grad_beta = grad_output.mean(axis=1, dtype=np.float64)
        grad_gamma = (grad_output*self._s_hat).mean(axis=1, dtype=np.float64)
        grad_output = grad_output * self.gamma[:,np.newaxis]

        self.beta -= eta* grad_beta # update scale param
        self.gamma -= eta * grad_gamma # update shift param
        
        ## Batch normalization
        sigma1 = (self._var_b_2 + self.eps)**-0.5
        sigma2 = (self._var_b_2 + self.eps)**-1.5
        G1 = grad_output * sigma1[:,np.newaxis]
        G2 = grad_output * sigma2[:,np.newaxis]
        D = X - self._mu_b[:,np.newaxis]
        c = (G2 * D).mean(1, dtype=np.float64)
        n = grad_output.shape[1]
        gradient_input = G1 - G1.mean(1, dtype=np.float64)[:, np.newaxis] - D*c[:, np.newaxis]
        return gradient_input

class Dropout(Layer):
    def __init__(self, p=0.5):
        # Drop out with droput probability p
        self.fc = 'DP' # type of layer
        self.p = p
        self.mask = None

    def forward(self, X, training = False):
        # Apply droput mask to [input_units, batch] matrix
        #np.random.seed(int(self.p*1000)) # uncomment for gradient check
        if training:
            self.mask = np.random.rand(X.shape[0], X.shape[1]) > self.p # keep
            X = X * self.mask / (1-self.p)
        return X

    def backward(self, X, grad_output):
        # Compute gradient of loss w.r.t. dropout
        masked_grad = self.mask / (1-self.p)
        return grad_output*masked_grad
###########################################

    
class Network():
    def __init__(self,layers = []):
        self.layers = layers[:]
        self.startup()
        
    def add(self,layer):
        self.layers.append(layer)
    
    def startup(self, n_epochs=40, n_batch=100, eta=0.001, reg=0.01, eta_min=None, eta_max=None, n_s=None, retrain=True, augment=False):
        """ Initializes the NN with the given params and empties the old logging lists. 
        """
        # Main param
        self.n_batch = n_batch
        self.eta = eta # learning rate
        self.reg = reg # regularization of weights
        
        ## Cyclic learning rate
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_s = n_s
        self.augment = augment
        
        if retrain:
            ## Logging
            self.t = 0 # nr of updates
            self.train_acc_log = []
            self.val_acc_log = []
            self.train_loss_log = []
            self.val_loss_log = []
            self.train_cost_log = []
            self.val_cost_log = []
            self.eta_log = []
        
    def re_init(self, init='He', std=None):
        """ Re-Initializes the weights of each layer of the NN
        """
        for l in self.layers:
            if l.fc=='FC':
                l.__init__(l.ins, l.outs, init, std)   
    
  #############################################    
  ########           Logging           ########                  
    def logging(self, X_train, Y_train_hot, X_valid, Y_valid_hot, visualize, epoch):
        """ Logs loss, cost and accuracy values of the training and validation set.
        """
        ll = 0
        for l in self.layers:
            if l.fc=='FC':
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
        """ Plotes the logged data.
        """
        fig, axs = plt.subplots(1,3, figsize=(15, 4))
        fig.subplots_adjust(hspace = .5, wspace=0.3)
        axs = axs.ravel()
        axs[0].plot(self.train_cost_log,label='train cost')
        axs[0].plot(self.val_cost_log,label='val cost')
        axs[0].legend(loc='best')
        axs[0].set_title('Cost function')
        axs[0].set_xlabel('epochs', fontsize=10)
        axs[0].grid()
        axs[1].plot(self.train_loss_log,label='train loss')
        axs[1].plot(self.val_loss_log,label='val loss')
        axs[1].legend(loc='best')
        axs[1].set_title('loss function')
        axs[1].set_xlabel('epochs', fontsize=10)
        axs[1].grid()
        axs[2].plot(self.train_acc_log,label='train accuracy')
        axs[2].plot(self.val_acc_log,label='val accuracy')
        axs[2].legend(loc='best')
        axs[2].set_title('Accuracy')
        axs[2].set_xlabel('epochs', fontsize=10)
        axs[2].grid()
        plt.show()
  #############################################
    
 
  #############################################    
  ######## Cross Entropy loss function ########    
    def CrossEntropyLoss(self, softmax_input, P, Y_batch):
        """ Calculates the gradient of the CrossEntropyLoss.

            Args:
                    softmax_input: input of the SoftMax(or output of the last layer)
                    P: SoftMax activations of softmax_input
                    Y_batch: one-hotcoded array of column vectors for the used minibatch
            Output:
                    lossgrad: Gradient of the CrossEntropyLoss with respect to the output of last layer.
        """
        lossgrad = -(Y_batch - P)
        tmp = np.exp(softmax_input)
        return lossgrad
    
    def SoftMax(self, output):
        """ Performs SoftMax activation on the output
        """
        tmp = np.exp(output)
        softmax = tmp / np.sum(tmp,0)[np.newaxis,:]
        return softmax
       
    def loss(self, X_batch, Y_batch, training=False):
        """ Calculates the loss.

            Args:
                    X_batch: an minibatch of size [D, Nbatch] where D is the dimension of the features
                    Y_batch: one-hotcoded array of column vectors for the minibatch
            Output:
                    loss: Loss  
        """
        activations = self.forward(X_batch,training)
        softmax_input = activations[-1]
        tmp = np.exp(softmax_input)
        loss_matrix = - softmax_input + np.log(np.sum(tmp,0))[np.newaxis,:]
        return (loss_matrix * Y_batch).sum()/Y_batch.shape[1]

    def cost(self, X_batch, Y_batch, reg=None, training=False):
        """ Performs SoftMax activation on the output
        
            Arg:
                    X_batch: an minibatch of size [D, Nbatch] where D is the dimension of the features
                    Y_batch: one-hotcoded array of column vectors for the minibatch
                    reg: if it is not none the cost will be computed using it as regularization param(lambda)
            Output:
                    cost: calculated for the given samples and their labels
        """
        reg_to_use = self.reg
        if reg is not None:
            reg_to_use = reg
            
        ll = 0
        for l in self.layers:
            if l.fc == 'FC':
                ll +=reg_to_use*np.sum(l.weights[:]**2)
                
        return self.loss(X_batch, Y_batch, training=training) + ll
  #############################################
    
    def forward(self, X, training=False):
        """ Performs a forward step without the SoftMax at the end

            Args:
                    X: an minibatch of size [D, Nbatch] where D is the dimension of the features
            Output:
                    activations: A list of activations after each layer. The activation after the last layer is left out.
        """
        tmp = X
        activations = []
        activations.append(tmp)
        for l in self.layers:
            if l.fc == 'BN' or l.fc == 'DP':
                tmp = l.forward(activations[-1],training) # Training is True it is training
            else:
                tmp = l.forward(activations[-1])
            activations.append(tmp)
        return activations
    
    def backward(self, X_batch, Y_batch):
        """ Does a backward step and updates the weights and biases too.
        """
        if Y_batch.ndim == 1:
            Y_batch = Y_batch.reshape(Y_batch.shape[0],1)
        activations = self.forward(X_batch, training=True)
        lossgrad = self.CrossEntropyLoss(activations[-1], self.SoftMax(activations[-1]), Y_batch)
        for i, l in reversed(list(enumerate(self.layers))):
            if l.fc =='FC': 
                lossgrad = l.backward(activations[i], lossgrad, self.eta, self.reg)
            elif l.fc == 'BN':
                lossgrad = l.backward(activations[i], lossgrad, self.eta)#batchnorm
            else: # ReLu # Droput
                lossgrad = l.backward(activations[i],lossgrad)
    
    def predict(self, X):
        """ Performs a classification inference
        """
        p = self.SoftMax(self.forward(X)[-1])
        return p.argmax(0)
    
    def accuracy(self, X, Y):
        """ Calculates the classification accuracy on X
        """
        Y_pred = self.predict(X)
        if Y.ndim == 2:
            Y = Y.argmax(0)
        return np.mean(Y_pred == Y)
    
    def cyclical_learning_rate(self):
        """ Updates the learning rate based on the momentan global update step and CLR parameters
        """
        if self.eta_min and self.eta_max and self.n_s:
            t = np.mod(self.t, 2*self.n_s)
            self.eta = self.eta_min + np.copysign(np.mod(t,self.n_s),self.n_s-t)*(self.eta_max- self.eta_min)/self.n_s + (self.n_s<=t)*(self.eta_max-self.eta_min)
            self.eta_log.append(self.eta)

    def minibatch_SGD(self, X_train, Y_train_hot):
        """ Does the updates for one minibatch
        """
        n = X_train.shape[1]
        for j in range(0,n, self.n_batch):
                    self.cyclical_learning_rate() # updates the learning rate
                    X_batch = X_train[:, j:j+self.n_batch]
                    Y_batch = Y_train_hot[:, j:j+self.n_batch]
                    if self.augment: X_batch = self.augment_batch(X_batch,std = 0.13)
                    self.backward(X_batch, Y_batch)
                    self.t += 1

    def augment_batch(self, X_batch,std = 0.1):
        """ Pixelwise augmentation of training samples with 0 mean and 0.1 std gaussian noise.
        """
        return X_batch + np.random.normal(scale = std, size=X_batch.shape) # variance = std^2
    
    def train(self, X_train, Y_train_hot, X_valid, Y_valid_hot, retrain = True,
              shufle=True, n_epochs=40, n_batch=100, eta=0.001, reg=0.01, visualize=False, 
              eta_min=None, eta_max=None, n_s=None, augment = False, ensamble_list=None, 
              ensamble_name ="ensamble", init ='He',std=None):
        
        """ Trains the NN from which it is called from. 

        Args:
               X_train, X_valid: training and validation matrixes given as [D x Ntr] and [D x Nval]arrays, 
                                 D dimension of features, N nr of samples 
               Y_train_hot, Y_valid_hot:  one-hotcoded labels given as  [C x N] where C is nr of classes
               retrain: is true if retraining should be performed and old weights and biases have to be reseted
               shufle: true if shuffling before each epoche is enabled
               n_epochs: nr of training epoches
               n_batch: batch size which decides how many samples are used to build the Stochastic Gradient
                        nr of updates per epoch can be calculated as Ntr/n_batch.
               eta: learning rate
               reg: regularization hyperparameter otherwise called lambda
               visualize: true if training should be visualized
               eta_min, eta_max, n_s: parameters for cyclical learning rate where the range is [eta_min, eta_max]
                                      and the step size is n_s, i.e a full cycle is done in 2*n_s updates. All three
                                      parameters have to be not None in order to use cyclical learning rate.
               augment: true if training samples have to be augmented pixelwise with 0 mean and 0.1 std
               ensamble_list, ensamble_name: only if both are not None an ensamble will be build where ensamble_list
                                             contains the epochs after which an ensamble member should be saved.
                                             ensamble_name is a name prefix under which the ensamble members will be saved in N                                              Network/ensamble_name+"epochnr"


        Example:
           nn.train(X_train, Y_train, X_val, Y_val, shufle = True, n_epochs=n_epochs, eta=0.001, reg = 0.000470, 
                    visualize=True, eta_min=1e-5, eta_max=1e-1,n_s=900, augment = False)
        """
        updates_epoch = X_train.shape[1]/n_batch

        # Delete old logging
        self.startup(n_epochs, n_batch, eta, reg, eta_min, eta_max, n_s, retrain, augment)
            
        # Re-init weighs
        if retrain: self.re_init(init,std)
        

        print(int(self.t/updates_epoch)+1)
        if self.t == 0: self.logging(X_train, Y_train_hot, X_valid, Y_valid_hot, visualize,0)
        for epoch in tqdm(range(n_epochs)):
            if shufle:
                X_train, Y_train_hot = shuffle(X_train,Y_train_hot) # shuffle

            ## Minibatch SGD
            self.minibatch_SGD(X_train, Y_train_hot)
            
            ## Logging
            self.logging(X_train, Y_train_hot, X_valid, Y_valid_hot, visualize, int(self.t/updates_epoch)+1)
        
            ## Ensamble
            if ensamble_list is not None:
                if int(self.t/updates_epoch)+1 in ensamble_list:
                    print("Save NN for ensamble in epoch:",int(self.t/updates_epoch)+1)
                    pickle.dump(self, open("Networks/"+ensamble_name+str(epoch+1)+".p", "wb"))
                    
                    
# =============================== Preproccessing =============================#
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

def one_hot(n_class,Y):
	""" Return one hot encoding of Y."""
	one_hot_targets = np.eye(n_class)[:,Y]
	return one_hot_targets

def preprocess(X):
	""" Centering and normalization of data where observations are rows of X. """
	mean = np.mean(X,0)
	std = np.std(X,0)
	X = X - mean[np.newaxis,:]
	X = X / std[np.newaxis,:]
	return X.T

def create_val_set(X,Y,n_val=5000):
    X_val = X[:,0:n_val]
    Y_val = Y[:,0:n_val]
    X = X[:,n_val:]
    Y = Y[:,n_val:]
    return X, Y, X_val, Y_val

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('../Dataset/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	Y = dict[b'labels']
	X = dict[b'data']
	filenames = dict[b'filenames']
	return X,Y,filenames
    
def load_all_and_preproc():
	files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
	X = []
	Y = []
	for f in files:
		X_temp,Y_temp,_ = LoadBatch(f)
		X_temp = preprocess(X_temp)
		if len(X) and len(Y):
			X = np.hstack((X,X_temp))
			Y = np.hstack((Y,Y_temp))
		else:
			X = X_temp
			Y = Y_temp
	return X,Y

def visualize_multiple_images(X,h,w,rand=False):
	""" Visualizes w*h images from X, where X is a  N*widht*height*3 matrix, i.e. has N images."""
	import matplotlib.pyplot as plt
	fig, axes1 = plt.subplots(h,w,figsize=(w,h))
	i = 0
	for j in range(h):
		for k in range(w):
			i +=1 
			if rand:
				i = np.random.choice(range(len(X)))
			if h == 1:
				axes1[k].set_axis_off()
				axes1[k].imshow(X[i],interpolation='linear')
			else:
				axes1[j][k].set_axis_off()
				axes1[j][k].imshow(X[i]) 

# ============================= CHECKING GRADIENTS =============================#
def ComputeGradsNumSlow(X, Y, nn, lamda=0, h=1e-6):
    NR = 0 
    """ Converted from matlab code 
    """
    print("CALCULATING NUMERICAL GRADIENTS")
    gradients = []
    for n in tqdm(range(len(nn.layers))):
        layer = nn.layers[n]
        if layer.fc == 'FC': # only if layer is fully connected
            dic = {} # new dictionary
            b_backup = np.copy(layer.biases) # take a copy of the biases as backup
            b_try = layer.biases # b_try and layer.biases are the same list now
            grad_b = np.zeros(layer.weights.shape[0],dtype=np.float64) #,dtype=np.float64
            grad_W = np.zeros(layer.weights.shape,dtype=np.float64)
            for i in range(len(layer.biases)):
                b_try[i] -= h # augment
                c1 = nn.cost(X, Y, lamda, training=True) # compute cost
                b_try[i] = b_backup[i] # set b back to its initial value
                b_try[i] += h
                c2 = nn.cost(X, Y, lamda, training=True)
                b_try[i] = b_backup[i]
                grad_b[i] = (c2-c1) / (2*h)
                
            W_backup = np.copy(layer.weights) # take a copy of weights as back up
            W_try = layer.weights # W_try and nn.weights are now the same list
            for i in range(layer.weights.shape[0]):
                for j in range(layer.weights.shape[1]):
                    W_try[i,j] -= h # augment
                    c1 = nn.cost(X, Y, lamda, training=True) # compute cost
                    W_try[i,j] = W_backup[i,j] # set W back to its initial value
                    W_try[i,j] += h
                    c2 = nn.cost(X, Y, lamda, training=True)
                    W_try[i,j] = W_backup[i,j] 
                    grad_W[i,j] = (c2-c1) / (2*h)
            dic['grad_W'] = grad_W
            dic['grad_b'] = grad_b
            gradients.append(dic)

        elif layer.fc == 'BN': # only if layer is fully connected
            dic = {} # new dictionary
            beta_backup = np.copy(layer.beta) # take a copy of the beta as backup
            beta_try = layer.beta # b_try and layer.beta are the same list now
            grad_beta = np.zeros(layer.beta.shape,dtype=np.float64)
            for i in range(len(layer.beta)):               
                beta_try[i] -= h # augment
                c1 = nn.cost(X, Y, lamda, training=True) # compute cost
                beta_try[i] = beta_backup[i] # set b back to its initial value
                beta_try[i] += h
                c2 = nn.cost(X, Y, lamda, training=True)
                beta_try[i] = beta_backup[i]
                grad_beta[i] = (c2-c1) / (2*h)
            
            gamma_backup = np.copy(layer.gamma) # take a copy of gamma as back up
            gamma_try = layer.gamma # W_try and nn.gamma are now the same list
            grad_gamma = np.zeros(layer.gamma.shape, dtype=np.float64)
            for i in range(len(layer.gamma)):               
                gamma_try[i] -= h # augment
                c1 = nn.cost(X, Y, lamda, training=True) # compute cost
                gamma_try[i] = gamma_backup[i] # set b back to its initial value
                gamma_try[i] += h
                c2 = nn.cost(X, Y, lamda, training=True)
                gamma_try[i] = gamma_backup[i]
                grad_gamma[i] = (c2-c1) / (2*h)
                
            dic['grad_beta'] = grad_beta
            dic['grad_gamma'] = grad_gamma
            gradients.append(dic)
    return gradients

def comp_gradients(nn, X_batch, Y_batch, lamda=0):
    """ Does a backward step and updates the weights and biases too.
    """
    nn.reg = lamda    
    if Y_batch.ndim == 1:
        Y_batch = Y_batch.reshape(Y_batch.shape[0],1)

    activations = nn.forward(X_batch,training=True)
    lossgrad = nn.CrossEntropyLoss(activations[-1], nn.SoftMax(activations[-1]), Y_batch)
    
    gradients = []
    passed_grads = []
    dicc = {}
    dicc['input'] =0
    dicc['grad'] = lossgrad
    passed_grads.append(dicc)
    for i, l in reversed(list(enumerate(nn.layers))):
        if l.fc == 'FC': ## FClayer
            grad_weights = np.dot(lossgrad, activations[i].T)/(activations[i].shape[1]) 
            if nn.reg: grad_weights += 2 * nn.reg * l.weights
            grad_biases = lossgrad.mean(axis=1, dtype=np.float64)
            
            # save gradients
            dic = {}
            dic['grad_W'] = grad_weights
            dic['grad_b'] = grad_biases
            gradients.append(dic)
            lossgrad = l.backward(activations[i], lossgrad, nn.eta, nn.reg)
        elif l.fc == 'BN':     ## BN gradients         
            ## Shift and scale
            grad_output = lossgrad * l.gamma[:,np.newaxis]
            grad_beta = grad_output.mean(axis=1, dtype=np.float64) # update scale param
            grad_gamma = (grad_output*l._s_hat).mean(axis=1, dtype=np.float64) # update shift param
            
	    # save gradiens
            dic = {}
            dic['grad_beta'] = grad_beta
            dic['grad_gamma'] = grad_gamma
            gradients.append(dic)
            lossgrad = l.backward(activations[i], lossgrad, nn.eta)      
        else:
            lossgrad = l.backward(activations[i],lossgrad)

        dicc = {}
        dicc['input'] = activations[i]
        dicc['grad'] = lossgrad
        passed_grads.append(dicc)
    gradients.reverse()  
    passed_grads.reverse()  
    return gradients, passed_grads, activations

def test_gradients(nn,X,Y,lamda):
    gradientsdic = ComputeGradsNumSlow(X[0:50,0:1200], Y[:,0:1200], nn, lamda=0.1, h=1e-6)
    gradientsdic2, passed_grads, act = comp_gradients(nn, X[0:50,0:1200], Y[:,0:1200], lamda=0.1)
    clear_output(wait=True)

    i = 0
    n_fc= 0
    for k,l in list(enumerate(nn.layers)):
        if l.fc == 'FC':
            grad_num = gradientsdic2[i]
            grad = gradientsdic[i]
            n_fc +=1
            print("LAYER: "+str(n_fc))
            maks_bias = np.max( np.abs(np.concatenate((grad_num['grad_b'][np.newaxis,:],grad['grad_b'][np.newaxis,:]), 
                                      axis = 0)),axis = 0)

            err_bias = np.nansum(np.abs(grad_num['grad_b'] - grad['grad_b'])/maks_bias)
            maks_weight = np.max(np.abs(np.concatenate((grad_num['grad_W'][np.newaxis,:,:],grad['grad_W'][np.newaxis,:,:]),
                                                       axis = 0)),axis = 0)
            err_weights = np.nansum(np.abs(grad_num['grad_W'] - grad['grad_W'])/maks_weight)
            print("Sum of relative weights error for Layer FC :",err_weights)
            if k+1 >= len(nn.layers) or nn.layers[k+1].fc != 'BN': 
                print("Sum of relative biases error for Layer FC :",err_bias)
        elif l.fc == 'BN':
            grad_num = gradientsdic2[i]
            grad = gradientsdic[i]

            maks_beta = np.max( np.abs(np.concatenate((grad_num['grad_beta'][np.newaxis,:],grad['grad_beta'][np.newaxis,:]), 
                                      axis = 0)),axis = 0)
            err_beta = np.nansum(np.abs(grad_num['grad_beta'] - grad['grad_beta'])/maks_beta)
            maks_gamma = np.max( np.abs(np.concatenate((grad_num['grad_gamma'][np.newaxis,:],grad['grad_gamma'][np.newaxis,:]), 
                                      axis = 0)),axis = 0)
            err_gamma = np.nansum(np.abs(grad_num['grad_gamma'] - grad['grad_gamma'])/maks_gamma)
            print("Sum of relative betas error for Layer BN: :",err_beta)
            print("Sum of relative gammas error for Layer BN: :",err_gamma)
            print()
        else:
            i -=1 # relu
        i +=1
