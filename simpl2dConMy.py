from turtle import forward
import numpy as np
import math
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

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


class SimpleConv2d():
    def __init__(self, F, C, FH, FW, P, S,initializer=None,optimizer=None,activation=None):
        self.P = P
        self.S = S
        self.initializer = initializer
        self.optimizer = optimizer
        self.activation = activation
        self.W = self.initializer.W(F,C,FH,FW)
        self.B = self.initializer.B(F)
    def output_shape2d(self,H,W,PH,PW,FH,FW,SH,SW):
        OH = (H +2*PH -FH)/SH +1
        OW = (W +2*PW -FW)/SW +1
        return int(OH),int(OW)
    
    def forward(self, X,debug=False):
        self.X = X
        N,C,H,W = self.X.shape
        
        F,C,FH,FW = self.W.shape
        OH,OW = self.output_shape2d(H,W,self.P,self.P,FH,FW,self.S,self.S)
        self.params = N,C,H,W,F,FH,FW,OH,OW
        A = np.zeros([N,F,OH,OW])
        self.X_pad = np.pad(self.X,((0,0),(0,0),(self.P,self.P),(self.P,self.P)))
        for n in range(N):
            for ch in range(F):
                for row in range(0,H,self.S):
                    for col in range(0,W,self.S):
                        if self.P == 0 and (W-2 <= col or H-2<=row):
                            continue
                        A[n,ch,row,col] = np.sum(self.X_pad[n,:,row:row+FH,col:col+FW]*self.W[ch,:,:,:]) +self.B[ch]
        if debug==True:
            return A
        else:
            return  self.activation.forward(A)
    
    def backward(self, dZ,debug=False):
        if debug==True:
            dA = dZ
        else:
            dA = self.activation.backward(dZ)
        N,C,H,W,F,FH,FW,OH,OW = self.params
        dZ = np.zeros(self.X_pad.shape)
        self.dW = np.zeros(self.W.shape)
        self.dB = np.zeros(self.B.shape)
        for n in range(N):
            for ch in range(F):
                for row in range(0,H,self.S):
                    for col in range(0,W,self.S):
                        if self.P == 0 and (W-2 <= col or H-2<=row):
                            continue
                        dZ[n,:,row:row+FH,col:col+FW] += dA[n,ch,row,col]*self.W[ch,:,:,:]
        if self.P == 0:
            dZ = np.delete(dZ,[0,H-1],axis=2)
            dZ = np.delete(dZ,[0,W-1],axis=3)
        else:
            dl_rows = range(self.P),range(H+self.P,H+2*self.P,1)
            dl_cols = range(self.P),range(W+self.P,W+2*self.P,1)
            dZ = np.delete(dZ,dl_rows,axis=2)
            dZ = np.delete(dZ,dl_cols,axis=3)
        for n in range(N):
            for ch in range(F):
                for row in range(OH):
                    for col in range(OW):
                        self.dW[ch,:,:,:] += dA[n,ch,row,col]*self.X_pad[n,:,row:row+FH,col:col+FW]
        for ch in range(F):
            self.dB[ch] = np.sum(dA[:,ch,:,:])
        self = self.optimizer.update(self)
        return dZ

class SimpleInitializerConv2d:
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def W(self, F, C, FH, FW):
        return self.sigma * np.random.randn(F,C,FH,FW)
    def B(self, F):
        return np.zeros(F)

class ReLU:
    def forward(self, A):
        self.A = A
        return np.clip(A, 0, None)
    def backward(self, dZ):
        return dZ * np.clip(np.sign(self.A), 0, None)
    
class SGD:
    def __init__(self, lr):
        self.lr = lr
    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return



class MaxPool2D():
    def __init__(self,P):
        self.P = P
        self.PA = None
        self.Pindex = None
    def forward(self,A):
        N,F,OH,OW = A.shape
        PH,PW = int(OH/self.P),int(OW/self.P)
        self.params = N,F,OH,OW,self.P,PH,PW
        self.PA = np.zeros([N,F,PH,PW])
        self.Pindex = np.zeros([N,F,PH,PW])
        for n in range(N):
            for ch in range(F):
                for row in range(PH):
                    for col in range(PW):
                        self.PA[n,ch,row,col] =np.max(A[n,ch,row*self.P:row*self.P+self.P,col*self.P:col*self.P+self.P])
                        self.Pindex[n,ch,row,col] = np.argmax(A[n,ch,row*self.P:row*self.P+self.P,col*self.P:col*self.P+self.P])
        return self.PA
    def backward(self,dA):
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

class Scratch2dCNNClassifier():
    def __init__(self, NN, CNN, n_epoch=5, n_batch=1, verbose = False):
        self.NN = NN
        self.CNN = CNN
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.verbose = verbose
        self.log_loss = np.zeros(self.n_epoch)
        self.log_acc = np.zeros(self.n_epoch)

    def loss_function(self,y,yt):
        delta = 1e-7
        return -np.mean(yt*np.log(y+delta))
    def accuracy(self,Z,Y):
        return accuracy_score(Y,Z)

    def fit(self, X, y, X_val=False, y_val=False):
        for epoch in range(self.n_epoch):
            get_mini_batch = GetMiniBatch(X, y, batch_size=self.n_batch)
            self.loss = 0
            for mini_X_train, mini_y_train in get_mini_batch:              
                forward_data = mini_X_train[:,np.newaxis,:,:]
                for layer in range(len(self.CNN)):
                    forward_data = self.CNN[layer].forward(forward_data)
                flt = Flatten()
                forward_data = flt.forward(forward_data)
                for layer in range(len(self.NN)):
                    forward_data = self.NN[layer].forward(forward_data)
                Z = forward_data
                backward_data = (Z - mini_y_train)/self.n_batch
                for layer in range(len(self.NN)-1,-1,-1):
                    backward_data = self.NN[layer].backward(backward_data)
                backward_data = flt.backward(backward_data)
                for layer in range(len(self.CNN)-1,-1,-1):
                    backward_data = self.CNN[layer].backward(backward_data)
                self.loss += self.loss_function(Z,mini_y_train)
                if self.verbose:
                    print('batch loss %f'%self.loss_function(Z,mini_y_train))
            if self.verbose:
                print(self.loss/len(get_mini_batch),self.accuracy(self.predict(X),np.argmax(y,axis=1)))
            self.log_loss[epoch] = self.loss/len(get_mini_batch)
            self.log_acc[epoch] = self.accuracy(self.predict(X),np.argmax(y,axis=1))

    def predict(self, X):
        pred_data = X[:,np.newaxis,:,:]
        for layer in range(len(self.CNN)):
            pred_data = self.CNN[layer].forward(pred_data)
        flt=Flatten()
        pred_data = flt.forward(pred_data)
        for layer in range(len(self.NN)):
            pred_data = self.NN[layer].forward(pred_data)
        return np.argmax(pred_data,axis=1)

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

class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.HW = 1
        self.HB = 1
    def update(self, layer):
        self.HW += layer.dW**2
        self.HB += layer.dB**2
        layer.W -= self.lr * np.sqrt(1/self.HW) * layer.dW
        layer.B -= self.lr * np.sqrt(1/self.HB) * layer.dB
class SimpleInitializer:
    def __init__(self, sigma):
        self.sigma = sigma
    def W(self, *shape):
        W = self.sigma * np.random.randn(*shape)
        return W
    def B(self, *shape):
        B = self.sigma * np.random.randn(*shape)
        return B
class HeInitializer():
    def W(self, n_nodes1, n_nodes2):
        self.sigma = math.sqrt(2 / n_nodes1)
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2)
        return W
    def B(self, n_nodes2):
        B = self.sigma * np.random.randn(n_nodes2)
        return B
class GetMiniBatch:
    def __init__(self, X, y, batch_size = 20, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self._X = X[shuffle_index]
        self._y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int64)
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
class Tanh:
    def forward(self, A):
        self.A = A
        return np.tanh(A)
    def backward(self, dZ):
        return dZ * (1 - (np.tanh(self.A))**2)


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
class Flatten():
    def __init__(self):
        pass
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(len(X),-1)
    def backward(self, X):
        return X.reshape(self.shape)

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
               [ 0., -1.,  0.]]],

              [[[ 0.,  0.,  0.],
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
"""
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



# (1,1,4,4)
x = np.array([[[[ 1,  2,  3,  4],
                [ 5,  6,  7,  8],
                [ 9, 10, 11, 12],
                [13, 14, 15, 16]]]])

# (2,3,3)
w = np.array([[[ 0.,  0.,  0.],
               [ 0.,  1.,  0.],
               [ 0., -1.,  0.]],

              [[ 0.,  0.,  0.],
               [ 0., -1.,  1.],
               [ 0.,  0.,  0.]]])
               """
simple_conv_2d = SimpleConv2d(F=2, C=1, FH=3, FW=3, P=0, S=1,initializer=SimpleInitializerConv2d(),optimizer=SGD(0.01),activation=ReLU())
simple_conv_2d.W = w
print(x.shape)
print(w.shape)
A = simple_conv_2d.forward(x,True)
print(A)

#da = np.array([[[[ -4,  -4], [ 10,  11]],[[  1,  -7],[  1, -11]]]])
delta = np.array([[[[ -4,  -4],
                   [ 10,  11]],

                  [[  1,  -7],
                   [  1, -11]]]])
dZ = simple_conv_2d.backward(delta,True)
print(dZ)


result =simple_conv_2d.output_shape2d(H=6,W=6,PH=0,PW=0,FH=3,FW=3,SH=1,SW=1)
print(result)

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
test_data = np.zeros([10,2,3,3])
flat = Flatten()
flat_forward = flat.forward(test_data)
flat_backward = flat.backward(flat_forward)
print("test data shape:", test_data.shape)
print("Flat forward shape:", flat_forward.shape)
print("Flat backward shape:", flat_backward.shape)
############### Problem 7 ###################
# Learning and estimation
"""
NN = {
    0:FC(1960, 200, HeInitializer(), AdaGrad(0.01), ReLU()),
    1:FC(200, 200, HeInitializer(), AdaGrad(0.01), ReLU()),
    2:FC(200, 10, SimpleInitializer(0.01), AdaGrad(0.01), Softmax()),
}
CNN = {
    0:SimpleConv2d(F=10, C=1, FH=3, FW=3, P=1, S=1,initializer=SimpleInitializerConv2d(),optimizer=SGD(0.01),activation=ReLU()),
    1:MaxPool2D(2),
}
""" 
NN = {0:FC(7840, 200, HeInitializer(), AdaGrad(0.01), ReLU()),
    1:FC(200, 200, HeInitializer(), AdaGrad(0.01), ReLU()),
    2:FC(200, 10, SimpleInitializer(0.01), AdaGrad(0.01), Softmax()),}

CNN = {0: SimpleConv2d(F=10, C=1,FH=3,FW=3,P=1,S=1,
            initializer=SimpleInitializerConv2d(0.01), optimizer=SGD(0.1), activation=ReLU()),}

cnn2d = Scratch2dCNNClassifier(NN=NN, CNN=CNN, n_epoch=10, n_batch=200, verbose = True)

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

