import numpy as np

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('../Dataset/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	Y = dict[b'labels']
	X = dict[b'data']
	filenames = dict[b'filenames']
	return X,Y,filenames

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]
	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h
	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]
	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c1) / (2*h)
	return [grad_W, grad_b]

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(name+'.mat',{name:b})

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

            
## Mathematical functions 
def one_hot(n_class,Y):
	""" Return one hot encoding of Y."""
	one_hot_targets = np.eye(n_class)[:,Y]
	return one_hot_targets


def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def preprocess(X):
	""" Centering and normalization of data where observations are rows of X. """
	mean = np.mean(X,0)
	std = np.std(X,0)
	X = X - mean[np.newaxis,:]
	X = X / std[np.newaxis,:]
	return X.T
    
    
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
        
        
    
    
    
    
    
    
