
import numpy as np
import sys
import os
from tqdm import tqdm
from scipy.special import softmax

with open("goblet_book.txt", "r") as f:
    book_data = f.read()

book_chars = {uniq for uniq in book_data}
eps = sys.float_info.epsilon

class RNN:
    def __init__(self, book_chars=book_chars, m=100, eta=0.1, seq_length=25):
        """[Initializes the RNN]

        Keyword Arguments:
            book_chars {[dic]} -- [dictionary of all letters, needed for hot-encoding.] (default: {book_chars})
            m {int} -- [dimension of the hidden state] (default: {100})
            eta {float} -- [description] (default: {0.1})
            seq_length {int} -- [the expected sequence length. Needed to initialize the gradients saved of Adagrad.] (default: {25})
        """
        self.m = m
        K =len(book_chars)
        self.eta = eta
        self.seq_length = seq_length
        self.b = np.zeros((m,1))
        self.c = np.zeros((K,1))
        self.U = np.random.random((m,K))*0.1
        self.W = np.random.random((m,m))*0.1
        self.V = np.random.random((K,m))*0.1
        self.h = None

        self.grad =self.AdaGrad(K, m)
        self.mapper = self.Mapp(book_chars)

    def reset_back_info(self):
        """[Resets the parameters used in Backprop for better ram utilization]
        """
        self.A = None
        self.H = None
        self.O = None

    def forward(self, X, h, training=False):
        """[ Does a forward pass on the minibatch X]
        
        Arguments:
            X {[KxN_batch]} -- [minibatch of hot encoded vectors]
            h {[mx1]} -- [initial hidden state]
        
        Returns:
            [KxN_batch] -- [array with discrete probability distributions as columns]
        """
        if h.ndim == 1: h = h.reshape((h.shape[0],1))
        if type(X)==str: print(X)
        if X.ndim == 1: X = X.reshape((X.shape[0],1))
        
        # Store for backpror
        self.A = np.zeros((self.W.shape[0], X.shape[1]))
        self.H = np.zeros((self.W.shape[0], X.shape[1]+1))
        self.O = np.zeros((self.V.shape[0], X.shape[1]))

        P = np.zeros((self.V.shape[0], X.shape[1]))

        self.H[:,[0]] = h
        for i,x in enumerate(X.T):            
            self.A[:,[i]] = self.W @ self.H[:,[i]] + self.U @ x[:,np.newaxis] + self.b
            self.H[:,[i+1]] = np.tanh(self.A[:,[i]])
            self.O[:,[i]] = self.V @ self.H[:,[i+1]] + self.c
            P[:,[i]] = softmax(self.O[:,[i]])
        if training: self.h = self.H[:,[-1]]
        return P
    
    def predict(self, x, h, n=1):
        """[ Predicts a sequence of n letters.]
        
        Arguments:
            x {[Kx1]} -- [first input vector to RNN]
            h {[mx1]} -- [initial hidden state]
            n {[int]} -- [lenth of sequence to be generated]

        Returns:
            [Kxn] -- [a word given as array with hot encoded columns]
        """
        
        if h.ndim == 1: h = h.reshape((h.shape[0],1))
        if type(x)==str: print(x)
        if x.ndim == 1: x = x.reshape((x.shape[0],1))

        res = np.zeros((x.shape[0],0)) #end of sequence
        for i in range(n):
            p= self.forward(x,h)
            h = self.H[:,[-1]]
            self.reset_back_info()

            # sampling
            w = np.cumsum(p,axis=0)
            rr = np.random.rand(w.shape[1])
            ind = np.argmax(w >= rr[np.newaxis,:], axis=0)
            x = self.mapper[ind] # x_next is an array with one-hot encoded vectors as columns
            res = np.hstack((res,x))
        return res

    def word(self, word):
        """[Maps array of hot-encoded vectors to word]

        Arguments:
            word {[KxN_length]} -- [array of hot-encoded cikznbs to be mapped]

        Returns:
            [] -- [Word]
        """
        return self.mapper[word]
    
    def backward(self, X, Y, h0):
        """[Performs a forward and a backward pass and updates the parameters according to the self.grad optimizer]

        Arguments:
            X {[KxN]} -- [Array of hot-encoded letters]
            Y {[KxN]} -- [Expected outputs]
            h0 {[mx1]} -- [Initial state]

        Returns:
            [double] -- [Loss for X,Y]
        """


        K,m,N = X.shape[0], h0.shape[0], X.shape[1]
        if self.h is not None: h0 = self.h

        #Forward pass
        P = self.forward(X,h0,True)
        G = -(Y-P)

        # Initialize gradiens. All gradients are rows beside G
        grad_a = np.zeros((N,m))
        grad_h = np.zeros((N,m)) #G.T @ self.V
        grad_h[[-1],:] = G[:,[-1]].T @ self.V 
        grad_a[[-1],:] = grad_h[[-1],:] * (1-self.H[:,-1]**2)

        for i in range(X.shape[1]-2,-1,-1):
            grad_h[[i],:] = G[:,[i]].T @ self.V + grad_a[[i+1],:] @ self.W
            grad_a[[i],:] = grad_h[[i],:] * (1-self.H[:,i+1]**2)

        grad_U = grad_a.T @ X.T 
        grad_W = grad_a.T @ self.H[:,:-1].T #
        grad_b = grad_a.sum(0,keepdims=True)
        grad_V = G @ self.H[:,1:].T
        grad_c = G.sum(1,keepdims=True) #

        # Gradient cutting
        grad_U = np.clip(grad_U, -5, 5)
        grad_W = np.clip(grad_W, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)

        # Update through adagrad
        self.grad.update(grad_U, grad_W, grad_b.T, grad_V, grad_c)
        self.grad.apply(self)

        # Calc loss
        tmp = np.exp(self.O)
        loss_matrix = -self.O + np.log(np.sum(tmp,0))[np.newaxis,:]
        loss = (loss_matrix * Y).sum()
        self.reset_back_info()
        return loss

    def loss(self,X, Y, h0):
        """[Computes loss for an array with hot encoded columns ]
        
        Arguments:
            X {[KxN_samples]} -- [input that has form of an array with hot encoded columns]
            Y {[KxN_samples]} -- [expected output for each input sample given as array with hot encoded columns]
            h0 {[mx1]} -- [initial state]
        
        Returns:
            [double] -- [loss L(x_1:τ , y_1:τ , Θ) = sum l_t = sum −log(y_t^T p_t )]
        """
        P = self.forward(X, h0)
        tmp = np.exp(self.O)
        loss_matrix = - self.O + np.log(np.sum(tmp,0))[np.newaxis,:]
        self.reset_back_info()
        return (loss_matrix * Y).sum()

    ## Classes
    class Mapp:
        def __init__(self, book_chars):
            """[Initializes the mapping class.]

            Arguments:
                book_chars {[dic]} -- [dictionary of all letters]
            """
            self.K =len(book_chars)
            self.char_to_ind = { char:i for char,i in zip(book_chars,np.identity(self.K))}
            self.ind_to_char = { i:char for char,i in zip(book_chars,range(self.K))}
            
        def __getitem__(self, key):
            """[A generic mapper, transforming chars and indexes to hot-encoded vectors and visa-versa.]

            Arguments:
                key {[str]} -- [returns array with hot-encoded columns corresponding to each letter in key]
                key {[int]} -- [returns its corresponding hot-encoded vector]
                key {[np.ndarray]} -- [If it is an array with hot-encoded columns it returns the corresponding word]
                                      [If it is an array of indexes it returns the corresponding array of hot-encoded vectors.]
            Returns:
                [] -- [Depending on the case it is either a string or an array with hot-encoded columns. ]
            """
            # returns list array of one hot-encoded vectors
            if type(key) == str: # key is char or list of chars(word)
                return np.array([self.char_to_ind[k] for k in key]).T
            
            if type(key) == int:#if key is an index
                print(key)
                return self.ind_to_char[key]
            
            # returns char or list of chars
            # if key is an one hot encoded vector or an array with hot-encoded columns
            if type(key) == np.ndarray and key.ndim==1: key = key.reshape((key.shape[0],1))
                
            if type(key) == np.ndarray and np.sum(key) == key.shape[1] and len(key)== self.K:
                return [self.ind_to_char[np.argmax(k)] for k in key.T]
                
            #if key is numpy list of indexes
            elif(type(key) == np.ndarray and (np.sum(key) != 1 or len(key)==1)): 
                return np.array([self.char_to_ind[self.ind_to_char[int(k)]] for k in key]).T
            

            print("Error, invalid key :",key, type(key), key[0])
            return("smth went wrong. Invalid key.")

    class AdaGrad:
        def __init__(self, K,m):
            """[AdaGrad initialization.]

            Arguments:
                K {[int]} -- [dimension of the output]
                m {[int]} -- [dimension of the hidden state]
            """
            self.grad_U = None
            self.grad_W = None
            self.grad_b = None
            self.grad_V = None
            self.grad_c = None

            self.grad_U_2 = np.zeros((m,K))
            self.grad_W_2 = np.zeros((m,m))
            self.grad_b_2 = np.zeros((m,1))
            self.grad_V_2 = np.zeros((K,m))
            self.grad_c_2 = np.zeros((K,1))

        def update(self, grad_U, grad_W, grad_b, grad_V, grad_c):
            """[Updates the states of Adagrad]

            Arguments:
                grad_U {[mxJ]} -- []
                grad_W {[mxm]} -- []
                grad_b {[mx1]} -- []
                grad_V {[Kxm]} -- []
                grad_c {[Kx1]} -- []
            """
            self.grad_U = grad_U
            self.grad_W = grad_W
            self.grad_b = grad_b
            self.grad_V = grad_V
            self.grad_c = grad_c
            
            self.grad_U_2 += grad_U**2
            self.grad_W_2 += grad_W**2
            self.grad_b_2 += grad_b**2
            self.grad_V_2 += grad_V**2
            self.grad_c_2 += grad_c**2
        
        def apply(self,rnn):
            """[Applies one AdaGrad based gradient descent step on the parameters of rnn]

            Arguments:
                rnn {[RNN]} -- [RNN which parameters have to be updated]
            """
            rnn.b -= rnn.eta * self.grad_b   / np.sqrt(self.grad_b_2 + eps)
            rnn.c -= rnn.eta * self.grad_c   / np.sqrt(self.grad_c_2 + eps)
            rnn.U -= rnn.eta * self.grad_U   / np.sqrt(self.grad_U_2 + eps)
            rnn.W -= rnn.eta * self.grad_W   / np.sqrt(self.grad_W_2 + eps)
            rnn.V -= rnn.eta * self.grad_V   / np.sqrt(self.grad_V_2 + eps)

# ============================= CHECKING GRADIENTS =============================#
def ComputeGradsNumSlow(rnn,X, Y, h0, h=1e-6):
    """[Computes the gradients for each parameter of the rnn based on the central difference formula.]

    Arguments:
        rnn {[RNN]} -- [RNN which gradients are to be computed]
        X {[KxN]} -- [input of the rnn as an array with hot-encoded columns]
        Y {[KxN]} -- [expected output of the rnn as an array with hot-encoded columns]
        h0 {[mx1]} -- [initial hidden state]

    Keyword Arguments:
        h {[double]} -- [accuracy for central difference formula] (default: {1e-6})

    Returns:
        [tuple] -- [gradients for each parameter of the rnn]
    """
    print("CALCULATING NUMERICAL GRADIENTS")
    ######### b
    grad_b = np.zeros(rnn.b.shape, dtype=np.float64)
    b_backup = np.copy(rnn.b)
    b_try = rnn.b # b_try and rnn.b are the same list now
    for i in range(len(rnn.b)):
        b_try[i] -= h # augment
        c1 = rnn.loss(X, Y, h0) # compute cost
        b_try[i] = b_backup[i] # set b back to its initial value
        b_try[i] += h
        c2 = rnn.loss(X, Y, h0)
        b_try[i] = b_backup[i]
        grad_b[i] = (c2-c1) / (2*h)

    ######### c
    grad_c = np.zeros(rnn.c.shape, dtype=np.float64)
    c_backup = np.copy(rnn.c)
    c_try = rnn.c
    for i in range(len(rnn.c)):
        c_try[i] -= h
        c1 = rnn.loss(X, Y, h0)
        c_try[i] = c_backup[i]
        c_try[i] += h
        c2 = rnn.loss(X, Y, h0)
        c_try[i] = c_backup[i]
        grad_c[i] = (c2-c1) / (2*h)

    # return grad_b, grad_c, None, None, None
    ######### W
    grad_W = np.zeros(rnn.W.shape, dtype=np.float64)
    W_backup = np.copy(rnn.W)
    W_try = rnn.W
    for i in range(rnn.W.shape[0]):
        for j in range(rnn.W.shape[1]):
            W_try[i,j] -= h
            c1 = rnn.loss(X, Y, h0)
            W_try[i,j] = W_backup[i,j]
            W_try[i,j] += h
            c2 = rnn.loss(X, Y, h0)
            W_try[i,j] = W_backup[i,j]
            grad_W[i,j] = (c2-c1) / (2*h)

    ######### U
    grad_U = np.zeros(rnn.U.shape, dtype=np.float64)
    U_backup = np.copy(rnn.U)
    U_try = rnn.U
    for i in range(rnn.U.shape[0]):
        for j in range(rnn.U.shape[1]):
            U_try[i,j] -= h
            c1 = rnn.loss(X, Y, h0)
            U_try[i,j] = U_backup[i,j]
            U_try[i,j] += h
            c2 = rnn.loss(X, Y, h0)
            U_try[i,j] = U_backup[i,j]
            grad_U[i,j] = (c2-c1) / (2*h)

    ######### V
    grad_V = np.zeros(rnn.V.shape, dtype=np.float64)
    V_backup = np.copy(rnn.V)
    V_try = rnn.V
    for i in range(rnn.V.shape[0]):
        for j in range(rnn.V.shape[1]):
            V_try[i,j] -= h
            c1 = rnn.loss(X, Y, h0)
            V_try[i,j] = V_backup[i,j]
            V_try[i,j] += h
            c2 = rnn.loss(X, Y, h0)
            V_try[i,j] = V_backup[i,j]
            grad_V[i,j] = (c2-c1) / (2*h)

    return grad_b, grad_c, grad_U, grad_W, grad_V

def CompGradients(rnn, X, Y, h0):
    """[Computes the gradients for each parameter of the rnn by doing an forward-backward pass]

    Arguments:
        rnn {[RNN]} -- [RNN which gradients are to be computed]
        X {[KxN]} -- [input of the rnn as an array with hot-encoded columns]
        Y {[KxN]} -- [expected output of the rnn as an array with hot-encoded columns]
        h0 {[mx1]} -- [initial hidden state]

    Returns:
        [tuple] -- [gradients for each parameter of the rnn]
    """
    K,m,N = X.shape[0], h0.shape[0], X.shape[1]
    
    P = rnn.forward(X,h0)
    #rnn.reset_back_info()
    # print(P.shape, X.shape,Y.shape)
    G = -(Y-P) # grad 0 with gradients as columns

    # all gradients are rows beside G
    grad_a = np.zeros((N,m))
    grad_h = np.zeros((N,m)) #G.T @ rnn.V
    grad_h[[-1],:] = G[:,[-1]].T @ rnn.V 
    grad_a[[-1],:] = grad_h[[-1],:] * (1-rnn.H[:,-1]**2)
    # grad_a[[-1],:] = grad_h[[-1],:] * (1-np.tanh(rnn.A[:,-1])**2)

    for i in range(X.shape[1]-2,-1,-1):
        grad_h[[i],:] = G[:,[i]].T @ rnn.V + grad_a[[i+1],:] @ rnn.W
        grad_a[[i],:] = grad_h[[i],:] * (1-rnn.H[:,i+1]**2)
        # grad_a[[i],:] = grad_h[[i],:] * (1-np.tanh(rnn.A[:,i])**2)

    #print(grad_h.shape)
    grad_U = grad_a.T @ X.T 
    grad_W = grad_a.T @ rnn.H[:,:-1].T #
    grad_b = grad_a.sum(0,keepdims=True)
    grad_V = G @ rnn.H[:,1:].T
    grad_c = G.sum(1,keepdims=True) #

    rnn.reset_back_info()
    return grad_b.T, grad_c, grad_U, grad_W, grad_V

def CheckGradients(m = 5):
    h0 = np.random.rand(5,1)
    rnn_check = RNN(m=5)
    seq_length = 25

    # words                              # one hot
    X_chars= book_data[:seq_length];     X = rnn_check.mapper[X_chars]
    Y_chars = book_data[1:seq_length+1]; Y = rnn_check.mapper[Y_chars]

    
    grad_b_, grad_c_, grad_U_, grad_W_, grad_V_ = CompGradients(rnn_check, X, Y, h0)
    grad_b, grad_c, grad_U, grad_W, grad_V = ComputeGradsNumSlow(rnn_check,X, Y, h0, 1e-4)

    ## grad_b
    maks_bias = np.max( np.abs(np.concatenate((grad_b[np.newaxis,:],grad_b_[np.newaxis,:]), 
                    axis = 0)),axis = 0)
    err_bias = np.nansum(np.abs(grad_b - grad_b_)/maks_bias)
    print("Sum of relative weights error for the b :",err_bias)

    ## grad_c
    maks_bias = np.max( np.abs(np.concatenate((grad_c[np.newaxis,:],grad_c_[np.newaxis,:]), 
                    axis = 0)),axis = 0)
    err_bias = np.nansum(np.abs(grad_c - grad_c_)/maks_bias)
    print("Sum of relative weights error for the c :",err_bias)

    #grad W
    maks_weight = np.max(np.abs(np.concatenate((grad_W[np.newaxis,:,:], grad_W_[np.newaxis,:,:]),
                                            axis = 0)),axis = 0)
    err_weights = np.nansum(np.abs(grad_W - grad_W_)/maks_weight)
    print("Sum of relative weights error for the W :",err_weights)

    #grad V
    maks_weight = np.max(np.abs(np.concatenate((grad_V[np.newaxis,:,:], grad_V_[np.newaxis,:,:]),
                                            axis = 0)),axis = 0)
    err_weights = np.nansum(np.abs(grad_V - grad_V_)/maks_weight)
    print("Sum of relative weights error for the V :",err_weights)

    #grad U
    maks_weight = np.max(np.abs(np.concatenate((grad_U[np.newaxis,:,:], grad_U_[np.newaxis,:,:]),
                                            axis = 0)),axis = 0)
    err_weights = np.nansum(np.abs(grad_U - grad_U_)/maks_weight)
    print("Sum of relative weights error for the U :",err_weights)
# ==============================================================================#

def TrainRnn(seq_length=25, output_file='outtt.txt', synthethize_1000_letters_text = True):
    # initialize
    rnn = RNN()
    # words                              # one hot
    X_chars= book_data[:seq_length];     X = rnn.mapper[X_chars]
    Y_chars = book_data[1:seq_length+1]; Y = rnn.mapper[Y_chars]
    # init hidden state
    h0 = np.zeros((100,1))
    loss = rnn.backward(X, Y, h0)
    print(loss)
    ll = 0

    # training loop
    for k in range(8):
        for i in tqdm(range(0,len(book_data)-seq_length, seq_length)):
                
            # words                                  # one hot
            X_chars= book_data[i:i+seq_length];      X = rnn.mapper[X_chars]
            Y_chars = book_data[i+1:i+seq_length+1]; Y = rnn.mapper[Y_chars]
            
            # display loss every 250th update
            if np.mod(ll,250)==0:
                print(loss) 
            
            # synthesize a text of length 250 letters every 500th update
            if np.mod(ll,500)==0:
                txt = rnn.predict(x=X[:,0], h=rnn.h, n=250)
                if np.mod(ll,1000)==0: 
                    with open('outtt.txt', 'a') as f:
                        print("\n*iter =*" +str(ll)+"*, smooth_loos=*"+str(loss)+"\n", file=f)
                        print("".join(rnn.word(txt)), file=f)
                print("**iter =**" +str(ll)+"**, smooth_loos=**"+str(loss)+"\n" )
                print("".join(rnn.word(txt)))
                
            # Reset init hidden state every new epoche
            if i == 0:
                rnn.h=None
            loss = 0.999*loss + 0.001*rnn.backward(X, Y, h0); ll +=1

    # Synthesize text
    if synthethize_1000_letters_text:
        X_chars= book_data[len(book_data)-25]
        a = rnn.predict(x=rnn.mapper["H"], h=rnn.h, n=2000)
        l = "".join(rnn.word(a))
        print(l)

    return rnn



def main():
    CheckGradients()
    TrainRnn()

if __name__ == "__main__":
        main()