from turtle import forward
import numpy as np
import math
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

############# User defined classes #################

class FC():
    """
    FC layers from number of nodes n_nodes1 to n_nodes2
    Parameters
    --------------
    n_nodes1 : int
        Number of nodes in the previous layer
    n_nodes2 : int
        Number of nodes in later layer
    initializer : Instances of initialization methods
    optimizer : Instances of optimization methods
    activation : Activation function

    Returns
    --------------
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer, activation):
        self.optimizer = optimizer
        self.activation = activation

        self.W = initializer.W(n_nodes1, n_nodes2)
        self.B = initializer.B(n_nodes2)

    def forward(self, X):
        """
        Forward
        Parameters
        ----------------
        X : ndarray shape with (batch_size, n_nodes1)
            Input
        Returns
        ----------------
        A : ndarray shape with (batch_size, n_nodes2)
            Output
        """
        self.X = X
        A = X @ self.W + self.B
        return self.activation.forward(A)
    
    def backward(self, dA):
        """
        Backward
        Parameters
        ----------------
        dA : ndarray shape with (batch_size, n_nodes2)
            The gradient flowed in from behind
        Returns
        ----------------
        dZ : ndarray shape with (batch_size, n_nodes1)
            forward slope
        """
        dA = self.activation.backward(dA)
        dZ = dA@self.W.T
        self.dB = np.sum(dA, axis=0)
        self.dW = self.X.T@dA
        self.optimizer.update(self)
        return dZ

class SimpleInitializerConv2d():
    """
    Initialization with Gaussian distribution
    Parameters
    ----------------
    sigma: float
        standard deviation of Gaussian distribution
    """
    def __init__(self, sigma = 0.01):
        self.sigma = sigma
    def W(self, *shape):
        """
        Initializing weights
        Parameters:
        F,C,FH,FW
        Returns
        --------------------
        W: weights
        """
        W = self.sigma * np.random.randn(*shape)
        return W
    def B(self, *shape):
        """
        Initializing bias
        Parameters:
        F
        Returns
        --------------------
        B: biases
        """
        B = self.sigma * np.random.randn(*shape)
        return B

class SimpleInitializer():
    def __init__(self, sigma):
        self.sigma = sigma
    def W(self, *shape):
        W = self.sigma * np.random.randn(*shape)
        return W
    def B(self, *shape):
        B = self.sigma * np.random.randn(*shape)
        return B

class XavierInitializer():
    def W(self, n_nodes1, n_nodes2):
        self.sigma = math.sqrt(1 / n_nodes1)
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2)
        return W
    def B(self, n_nodes2):
        B = self.sigma * np.random.randn(n_nodes2)
        return B
    
class HeInitializer():
    def W(self, n_nodes1, n_nodes2):
        self.sigma = math.sqrt(2 / n_nodes1)
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2)
        return W
    def B(self, n_nodes2):
        B = self.sigma * np.random.randn(n_nodes2)
        return B

class SGD():
    def __init__(self, lr):
        self.lr = lr
    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return layer

class AdaGrad():
    def __init__(self, lr):
        self.lr = lr
        self.HW = 1
        self.HB = 1
    def update(self, layer):
        self.HW += layer.dW**2
        self.HB += layer.dB**2
        layer.W -= self.lr * np.sqrt(1/self.HW) * layer.dW
        layer.B -= self.lr * np.sqrt(1/self.HB) * layer.dB


class Sigmoid():
    def __init__(self):
        pass
    def forward(self, A):
        self.A = A
        return self.sigmoid(A)
    def backward(self, dZ):
        _sig = self.sigmoid(self.A)
        return dZ * (1 - _sig)*_sig
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

class Tanh():
    def __init__(self):
        pass
    def forward(self, A):
        self.A = A
        return np.tanh(A)
    def backward(self, dZ):
        return dZ * (1 - (np.tanh(self.A))**2)

class Softmax():
    def __init__(self):
        pass
    def forward(self, X):
        self.Z = np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1,1)
        return self.Z
    def backward(self, Y):
        self.loss = self.loss_func(Y)
        return self.Z - Y
    def loss_func(self, Y, Z=None):
        if Z is None:
            Z = self.Z
        return (-1)*np.average(np.sum(Y*np.log(Z), axis=1))

class ReLU():
    def __init__(self):
        pass
    def forward(self, A):
        self.A = A
        return np.maximum(self.A, 0)
    def backward(self, dZ):
        return np.where(self.A>0,dZ,0)

class GetMiniBatch:
    def __init__(self, X, y, batch_size = 20, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self._X = X[shuffle_index]
        self._y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)
    def __len__(self):
        return self._stop
    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self._X[p0:p1], self._y[p0:p1] 
    def __iter__(self):
        self._counter = 0
        return self
    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self._X[p0:p1], self._y[p0:p1]

################# Problem 1 ################

class SimpleConv2d():
    """
    Implementation of simple 2d convolution
    Parameters
    -------------
    Initializer : Instances of initialization methods
    Optimizer : Instances of optimization methods 
    Returns 
    -------------
    """
    def __init__(self, F, C, FH, FW, P, S,initializer=None,optimizer=None,activation=None):
        self.P = P
        self.S = S
        self.initializer = initializer
        self.optimizer = optimizer
        self.activation = activation

        self.W = self.initializer.W(F,C,FH,FW)
        self.B = self.initializer.B(F)
    
    def output_shape2d(self,H,W,PH,PW,FH,FW,SH,SW):
        OH = (H + 2 * PH - FH)/SH + 1
        OW = (W + 2 * PW - FW)/SW + 1
        return int(OH),int(OW)
    
    def forward(self, X):
        """
        forward
        Parameters
        -------------------
        X : ndarray shape with (batch_size, n_nodes1)
            Input

        Returns
        A : ndarray shape with (batch_size, n_nodes2)
            Output
        -------------------
        """
        self.X = X
        N,C,H,W  = self.X.shape
        F,C,FH,FW = self.W.shape

        OH, OW = self.output_shape2d(H,W,self.P, self.P, FH, FW, self.S, self.S)
        self.params = N,C,H,W,F,FH,FW,OH,OW

        A = np.zeros([N,F,OH,OW])

        self.X_pad = np.pad(self.X,((0,0),(0,0),(self.P, self.P), (self.P,self.P)))


        for n in range(N):
            for ch in range(F):
                for row in range(0,H,self.S):
                    for col in range(0,W,self.S):
                        A[n,ch,row,col] = np.sum(self.X_pad[n,:,row:row+FH, col:col+FW]*self.W[ch,:,:,:])+self.B[ch]
        
        return self.activation.forward(A)
       

    def backward(self, dA):
        """
        Backward
        Parameters
        ----------------
        dA : ndarray shape with (batch_size, n_nodes2)
            The gradient flowed in from behind
        Returns
        ----------------
        dZ : ndarray shape with (batch_size, n_nodes1)
            forward slope
        """

        dA = self.activation.backward(dA)
        N,C,H,W,F,FH,FW,OH,OW = self.params

        dZ = np.zeros(self.X_pad.shape)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)

        # dZ
        # Batch
        for n in range(N):
            for ch in range(F):
                for row in range(0,H,self.S):
                    for col in range(0,W, self.S):
                        dZ[n,:,row:row+FH, col:col+FW] += dA[n,ch,row,col]*self.W[ch,:,:,:]
        
        d1_rows = range(self.P), range(H+self.P, H+2*self.P,1)
        d1_cols = range(self.P), range(W+self.P, W+2*self.P,1)

        dZ = np.delete(dZ, d1_rows, axis =2)
        dZ = np.delete(dZ,d1_cols, axis = 3)

        # dW
        # Batch
        for n in range(N):
            for ch in range(F):
                for row in range(OH):
                    for col in range(OW):
                        self.dW[ch,:,:,:] += dA[n,ch,row,col]*self.X_pad[n,:,row:row+FH, col:col+FW]
        
        # dB
        # Out channel

        for ch in range(F):
            self.B[ch] = np.sum(dA[:,ch,:,:])
        
        # Update

        self = self.optimizer.update(self)

        return dZ   

################# Problem 4 ######################
# Creation of maximum pooling layer
class MaxPool2D():
    '''
    Perform max pooling
    Parameters
    --------------------
    P : int 
        max pooling size
    '''
    def __init__(self, P):
        self.P = P
        self.PA = None
        self.Pindex = None
    
    def forward(self, A):
        """
        forward
        Parameters
        -------------------
        A : ndarray shape with(n_batch, filter, height and width)
            training samples
        
        """
        N,F,OH,OW = A.shape
        PS = self.P
        PH,PW = int(OH/PS), int(OW/PS)

        self.params = N,F,OH,OW,PS,PH,PW

        # pooling filter
        self.PA = np.zeros([N,F,PH,PW])
        self.Pindex = np.zeros([N,F,PH,PW])

        for n in range(N):
            for ch in range(F):
                for row in range(PH):
                    for col in range(PW):
                        self.PA[n,ch,row,col] = np.max(A[n,ch,row*PS:row*PS+PS,col*PS:col*PS+PS])
                        self.Pindex[n,ch,row,col] = np.argmax(A[n,ch,row*PS:row*PS+PS,col*PS:col*PS+PS])

        return self.PA
    
    def backward(self, dA):
        N,F,OH,OW,PS,PH,PW = self.params
        dP = np.zeros([N,F,OH,OW])

        for n in range(N):
            for ch in range(F):
                for row in range(PH):
                    for col in range(PW):
                        idx = self.Pindex[n,ch,row,col]
                        tmp = np.zeros((PS*PS))
                        for i in range(PS*PS):
                            if i == idx:
                                tmp[i] = dA[n,ch,row,col]
                            else:
                                tmp[i] = 0
                        dP[n,ch,row*PS:row*PS+PS,col*PS:col*PS+PS] = tmp.reshape(PS,PS)
        return dP
############### Problem 5 ################
# (Advance task) Creating an average pooling
class AvgPool2D():
    '''
    Perform average pooling
    Parameters
    --------------------
    P : int 
         average pooling size
    '''
    def __init__(self, P):
        self.P = P
        self.PA = None
        self.Pindex = None
    
    def forward(self, A):
        """
        forward
        Parameters
        -------------------
        A : ndarray shape with(n_batch, filter, height and width)
            training samples
        
        """
        N,F,OH,OW = A.shape
        PS = self.P
        PH,PW = int(OH/PS), int(OW/PS)

        self.params = N,F,OH,OW,PS,PH,PW

        # pooling filter
        self.PA = np.zeros([N,F,PH,PW])

        for n in range(N):
            for ch in range(F):
                for row in range(PH):
                    for col in range(PW):
                        self.PA[n,ch,row,col] = np.mean(A[n,ch,row*PS:row*PS+PS,col*PS:col*PS+PS])

        return self.PA
    
    def backward(self, dA):
        N,F,OH,OW,PS,PH,PW = self.params
        dP = np.zeros([N,F,OH,OW])

        for n in range(N):
            for ch in range(F):
                for row in range(PH):
                    for col in range(PW):
                        tmp = np.zeros((PS*PS))
                        for i in range(PS*PS):
                                tmp[i] = dA[n,ch,row,col]/(PS*PS)
                        dP[n,ch,row*PS:row*PS+PS,col*PS:col*PS+PS] = tmp.reshape(PS,PS)
        return dP
############### Problem 6 ##################
# Smoothing
class flatten():
    def __init__(self):
        pass
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(len(X),-1)
    def backward(self, X):
        return X.reshape(self.shape)

############## Problem 7 ###################
class Scratch2dCNNClassifier:
    """
    2d conv layer 
    """
    def __init__(self, NN, CNN, n_epoch = 10, n_batch = 5, verbose = False):
        
        self.n_batch = n_batch
        self.n_epoch = n_epoch
        self.verbose = verbose

        self.log_loss = np.zeros(self.n_epoch)
        self.log_acc = np.zeros(self.n_epoch)
        self.NN = NN
        self.CNN = CNN
    def loss_function(self, y, yt):
        delta = 1e-7
        temp = -np.mean(yt*np.log(y + delta))
        return temp
    
    def accuracy(self, y, yt):
        return accuracy_score(y,yt)
    
    def fit(self, X, y, X_val = False, y_val = False):
        """
        Train a cnn classifier

        Parameters
        ---------------
        X : ndarray shape with (n_samples, n_features)
            features of training data
        y : ndarray shape with (n_samples, )
            True label of training data
        X_val : ndarray shape with (n_samples, n_features)
            features of validation data
        y_val : ndarray shape with (n_samples, )
            True label of validation data
        """

        for epoch in range(self.n_epoch):
            self.loss = 0
            get_mini_batch = GetMiniBatch(X,y, batch_size=self.n_batch)
            for mini_X_train, mini_y_train in get_mini_batch:

                forward_data = mini_X_train[:,np.newaxis,:,:]
                # conv layer
                for layer in range(len(self.CNN)):
                    forward_data = self.CNN[layer].forward(forward_data)

                #flatten layer
                flat = flatten()
                forward_data = flat.forward(forward_data)
                # NN
                for layer in range(len(self.NN)):
                    forward_data = self.NN[layer].forward(forward_data)
                
                Z = forward_data

                backward_data = (Z - mini_y_train)/self.n_batch
                for layer in range(len(self.NN)-1,-1,-1):
                    backward_data = self.NN[layer].backward(backward_data)

                backward_data = flat.backward(backward_data)

                for layer in range(len(self.CNN)-1,-1,-1):
                    backward_data = self.CNN[layer].backward(backward_data)
                
                self.loss += self.loss_function(Z, mini_y_train)

            self.log_loss[epoch] = self.loss/len(get_mini_batch)
            self.log_acc[epoch] = self.accuracy(self.predict(X), np.argmax(y, axis = 1))

            if self.verbose:
                print('epoch:{} loss:{} acc:{}'.format(epoch, self.loss/self.n_batch, self.log_acc[epoch]))   

    def predict(self, X):
        """
        Estimate using a neural network classifier
        
        Parameters
        ---------------
        X : ndarray shape with (n_samples, n_features)
            sample of dataset

        Returns
        ---------------
        pred : ndarray (n_samples, 1)
        """
        pred_data = X[:, np.newaxis, :,:]

        for layer in range(len(self.CNN)):
            pred_data = self.CNN[layer].forward(pred_data)
        
        flt=flatten()
        pred_data = flt.forward(pred_data)

        for layer in range(len(self.NN)):
            pred_data = self.NN[layer].forward(pred_data)
        
        pred = np.argmax(pred_data, axis = 1)
        return pred


################# Data set preparation ###############

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train data shape: ", X_train.shape) # (60000, 28, 28)
print("X_test data shape: ", X_test.shape) # (10000, 28, 28)

# Preprocessing
X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
X_train /= 255
X_test /= 255

# the correct label is an integer from 0 to 9, but it is converted to a one-hot representation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train_one_hot = enc.fit_transform(y_train.reshape(-1,1))
y_val_one_hot = enc.fit_transform(y_val.reshape(-1,1))

print("Train dataset:", X_train.shape) # (48000, 784)
print("Validation dataset:", X_val.shape) # (12000, 784)

############### Problem 2 & 3 ################
# Experiment of a two-dimensional convolution layer with a small array
def output_shape2d(H,W,PH,PW,FH,FW,SH,SW):
    OH = (H +2*PH -FH)/SH +1
    OW = (W +2*PW -FW)/SW +1
    return int(OH),int(OW)

x = np.array([[[[ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16]]]])

w = np.array([[[[ 0.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0., -1.,  0.]],

              [[ 0.,  0.,  0.],
               [ 0., -1.,  1.],
               [ 0.,  0.,  0.]]]])

#w = np.array([[[[ 0.,  0.,  0.],
#               [ 0.,  1.,  0.],
#               [ 0., -1.,  0.]]]])

#w = w[:,np.newaxis,:,:]
N,C,H,W = x.shape
F,C,FH,FW = w.shape
S = 1
P = 1
#w = np.ones([F,C,FH,FW])
b = np.ones((C,F))
print("x shape:", x.shape)
print("w shape", w.shape)
#print(w)

OH,OW = output_shape2d(H,W,P,P,FH,FW,S,S)
X_pad = np.pad(x,((0,0),(0,0),(P,P),(P,P)))
print("x pad:", X_pad)
#### forward ####
A = np.zeros([N,C,OH,OW])
for n in range(N):
    for ch in range(C):
        for row in range(0,H,S):
            for col in range(0,W,S):
                A[n,ch,row,col] = np.sum(X_pad[n,:,row:row+FH, col:col+FW] * w[:,ch,:,:]) + b[ch]
print("A shape:",A.shape)
print("A:", A)
print("X_pad shape:", X_pad.shape)
#### Backward


dA = np.ones(A.shape)
dZ = np.zeros(X_pad.shape)
dw = np.zeros(w.shape)
db = np.zeros(b.shape)

# dZ batch
for n in range(N):
    for ch in range(C):
        for row in range(0,H,S):
            for col in range(0,W,S):
                dZ[n,:,row:row+FH, col:col+FW] += dA[n,ch,row,col]*w[:,ch,:,:]


d1_rows = range(P), range(H+P, H+2*P,1)
d1_cols = range(P), range(W+P, W+2*P,1)

dZ = np.delete(dZ, d1_rows, axis =2)
dZ = np.delete(dZ,d1_cols, axis = 3)

# dW Batch
for n in range(N):
    for ch in range(C):
        for row in range(OH):
            for col in range(OW):
                w[:,ch,:,:] += dA[n,ch,row,col]*X_pad[n,:,row:row+FH, col:col+FW]
        
# dB Out channel

for ch in range(C):
    db[ch] = np.sum(dA[:,ch,:,:])
        
print("dZ:",dZ)
print("dW:", dw)
print("db:", db)

################ Problem 4 test ##################
test_data = np.random.randint(0,9,36).reshape(1,1,6,6)
maxpooling = MaxPool2D(P=2)
pool_forward = maxpooling.forward(test_data)
print("test data:", test_data)
print("Maxpooling forward:", pool_forward)
################ Problem 5 test ##################
test_data = np.random.randint(0,9,36).reshape(1,1,6,6)
avgpooling = AvgPool2D(P=2)
pool_forward = avgpooling.forward(test_data)
print("test data:", test_data)
print("Avgpooling forward:", pool_forward)
################ Problem 6 test ##################
test_data = np.zeros([10,2,5,5])
flat = flatten()
flat_forward = flat.forward(test_data)
flat_backward = flat.backward(flat_forward)
print("test data shape:", test_data.shape)
print("Flat forward shape:", flat_forward.shape)
print("Flat backward shape:", flat_backward.shape)
############### Problem 7 ###################
# Learning and estimation

NN = {0: FC(7840, 400, HeInitializer(), AdaGrad(0.01), Tanh()),
    1: FC(400, 200, HeInitializer(), AdaGrad(0.01), Tanh()),
    2: FC(200, 10, SimpleInitializer(0.01), AdaGrad(0.01), Softmax()),}

CNN = {0: SimpleConv2d(F=10, C=1,FH=3,FW=3,P=1,S=1,
            initializer=SimpleInitializerConv2d(0.01), optimizer=SGD(0.01), activation=ReLU()),}

cnn2d = Scratch2dCNNClassifier(NN=NN, CNN=CNN, n_epoch=10, n_batch=200, verbose = True)
cnn2d.fit(X_train[0:1000], y_train_one_hot[0:1000])

y_pred = cnn2d.predict(X_val[0:500])
acc = accuracy_score(y_val[0:500], y_pred)
print("Accuracy:", acc)

############# Problem 8 #################
#  (advanced task) LeNet

LeNetCNN = {0: SimpleConv2d(F=6, C=1,FH=5,FW=5,P=2,S=1,
            initializer=SimpleInitializerConv2d(0.01), optimizer=SGD(0.1), activation=ReLU()),
            1: MaxPool2D(P=2),
            2: SimpleConv2d(F=16, C=6,FH=5,FW=5,P=2,S=1,
            initializer=SimpleInitializerConv2d(0.01), optimizer=SGD(0.1), activation=ReLU()),
            3: MaxPool2D(P=2),}

LeNetNN = {0: FC(784, 120, HeInitializer(), AdaGrad(0.01), Tanh()),
    1: FC(120, 84, HeInitializer(), AdaGrad(0.01), Tanh()),
    2: FC(84, 10, SimpleInitializer(0.01), AdaGrad(0.01), Softmax()),}

LeNet = Scratch2dCNNClassifier(NN = LeNetNN, CNN = LeNetCNN, n_epoch=10, n_batch=100, verbose=True)
LeNet.fit(X_train[0:1000], y_train_one_hot[0:1000])

y_pred_lenet = LeNet.predict(X_val[0:500])
acc_lenet = accuracy_score(y_val[0:500], y_pred_lenet)
print("Accuracy:", acc_lenet)
############## Problem 10 ##############
# Calculation of output size and number of parameters
print("Parameters in general are weights that are learnt during training. Parameters can calculate using following formula:\n\
    (filter width*filter height*number of filter in the previous layer +1)* number of filters")
print("Example 1:")
print("input size: 144x144, 3")
print("Filter size: 3x3, 6")
print("Stride: 1")
print("Padding: None")
print("Number of parameter: 168")
print("output size: 142x142x6")

print("Example 2:")
print("input size: 60x60, 24")
print("Filter size: 3x3, 48")
print("Stride: 1")
print("Padding: None")
print("Number of parameter: 10416")
print("output size: 58x58x48")

print("Example 3:")
print("input size: 20x20, 10")
print("Filter size: 3x3, 20")
print("Stride: 2")
print("Padding: None")
print("Number of parameter: 1820")
print("output size: 9x9x20")