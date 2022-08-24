from platform import release
from re import L
import numpy as np
import math
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class Sigmoid:
    def forward(self, A):
        self.A = A
        return self.sigmoid(A)
    def backward(self, dZ):
        _sig = self.sigmoid(self.A)
        return dZ * (1 - _sig)*_sig
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

class Tanh:
    def forward(self, A):
        self.A = A
        return np.tanh(A)
    def backward(self, dZ):
        return dZ * (1 - (np.tanh(self.A))**2)

class Softmax:
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

class ReLU:
    def forward(self, A):
        self.A = A
        return np.clip(A, 0, None)
    def backward(self, dZ):
        return dZ * np.clip(np.sign(self.A), 0, None)

class FC:
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer,activation):
        self.optimizer = optimizer
        self.W = initializer.W(n_nodes1, n_nodes2)
        self.B = initializer.B(n_nodes2)
        self.activation = activation
    def forward(self, X):
        self.X = X
        A = X@self.W + self.B
        return self.activation.forward(A)
    def backward(self, dA):
        dA = self.activation.backward(dA)
        dZ = dA@self.W.T
        self.dB = np.sum(dA, axis=0)
        self.dW = self.X.T@dA
        self.optimizer.update(self)
        return dZ

class XavierInitializer:
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
        
class SimpleInitializer:
    def __init__(self, sigma):
        self.sigma = sigma
    def W(self, *shape):
        W = self.sigma * np.random.randn(*shape)
        return W
    def B(self, *shape):
        B = self.sigma * np.random.randn(*shape)
        return B

class SimpleInitializerConv1d:
    def __init__(self, sigma):
        self.sigma = sigma
    def W(self, *shape):
        W = self.sigma * np.random.randn(*shape)
        return W
    def B(self, *shape):
        B = self.sigma * np.random.randn(*shape)
        return B

class SGD:
    def __init__(self, lr):
        self.lr = lr
    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return

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


########### 1 #################
#Creation of a one-dimensional convolution layer class with the number of channels limited to one

class SimpleConv1d():
    '''
    1d conv layer
    Parameters
    -----------------
    n_nodes1 : int
        Number of nodes in the previous layer
    n_nodes 2 : int
        Number of nodes in later layers
    Initializer : Instances of initialization method
    Optimizer : Instances of optimization method
    '''
    def __init__(self, output_channel, input_channel, filter_size, padding = 0, stride = 1, initializer = None, optimizer = None, activation = None):
        self.initializer = initializer
        self.optimizer = optimizer 
        self.activation = activation

        self.W = self.initializer.W(output_channel, input_channel, filter_size)
        self.B = self.initializer.B(output_channel)

    def output_size_calculation(self, n_in, filter_size, padding=0, stride=1):
        """
        Calculate output size after 1d convolution

        Parameters
        -----------------
        n_in: Input size   
        F: filter size 
        P: padding number 
        S: stride number 

        Return
        -----------------
        n_out: size of output
        """
        n_out = int((n_in + 2*padding - filter_size) / stride + 1)   
        return n_out

    def forward(self, X):
        '''
        Calculate forward propagation
        Parameters
        --------------
        x : ndarray shape with (batch_size, n_nodes1)
            training feature

        returns
        ---------------
        A : ndarray shape with (batch_size, n_nodes2)
        '''
        self.X = X
        N,INC, Feature = X.shape
        OCH, INC, FS = self.W.shape
        OUT = self.output_size_calculation(Feature, FS, 0,1)
        self.size = N,INC,OCH,FS,OUT
        A = np.zeros([N,OCH,OUT])

        for n in range(N):
            for och in range(OCH):
                for ich in range(INC):
                    for m in range(OUT):
                        A[n,och,m] += np.sum(X[n, ich,m:m+FS]*self.W[och,ich,:]) 
        
        A += self.B[:,None]
        A = self.activation.forward(A)

        return A

    def backward(self, dZ):
        '''
        Calculate backward propagation
        Parameters
        --------------
        x: ndarray
            training feature
        w: ndarray
            weight
        da: ndarray
            backpropagation value
        '''
        dA = self.activation.backward(dZ)
        self.dB = np.mean(np.sum(dA, axis=2), axis = 0)

        self.dW = np.zeros(self.W.shape)
        dZ = np.zeros(self.X.shape)

        N,INC,OCH,FS,OUT = self.size
        for n in range(N):
            for och in range(OCH):
                for ich in range(INC):
                    for fs in range(FS):
                        for m in range(OUT):
                            self.dW[och,ich,fs] += self.X[n,ich,fs+m]*dA[n,och,m]
                            dZ[n,ich,fs+m] += self.W[och,ich,fs]*dA[n,och,m]
        
        self = self.optimizer.update(self)

        return dZ

class Scratch1dCNNClassifier:
    """
    1d conv layer 
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
                forward_data = mini_X_train.reshape(self.n_batch,1,-1)
                for layer in range(len(self.CNN)):
                    forward_data = self.CNN[layer].forward(forward_data)

                record_shape = forward_data.shape
                forward_data = forward_data.reshape(self.n_batch, -1)

                for layer in range(len(self.NN)):
                    forward_data = self.NN[layer].forward(forward_data)
                
                Z = forward_data

                backward_data = (Z - mini_y_train)/self.n_batch
                for layer in range(len(self.NN)-1,-1,-1):
                    backward_data = self.NN[layer].backward(backward_data)

                backward_data = backward_data.reshape(record_shape)

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
        pred_data = X[:, np.newaxis, :]

        for layer in range(len(self.CNN)):
            pred_data = self.CNN[layer].forward(pred_data)
        
        pred_data = pred_data.reshape(len(X),-1)
        for layer in range(len(self.CNN)):
            pred_data = self.NN[layer].forward(pred_data)
        
        pred = np.argmax(pred_data, axis = 1)
        return pred

################## 2 ##############

def output_size_calculation( n_in, filter_size, padding=0, stride=1):
        """
        Calculate output size after 1d convolution

        Parameters
        -----------------
        n_in: Input size   
        F: filter size 
        P: padding number 
        S: stride number 

        Return
        -----------------
        n_out: size of output
        """
        n_out = int((n_in + 2*padding - filter_size) / stride + 1)   
        return n_out

a = output_size_calculation(4,3,0,1)
print("output:", a)

################## 3 ##############
X = np.array([1,2,3,4])
w = np.array([3,5,7])
b = np.array([1])

# forward propagation
a = np.zeros(a)
for i in range(len(a)):
    x_temp = X[i:i+len(w)]
    a[i] = np.sum(x_temp*w)+b
print(a)

#  back propagation 
delta_a = np.array([10,20])
delta_b = np.sum(delta_a)
print(delta_b)

delta_w = np.zeros([len(w)])
for i in range(len(w)):
    x_temp = X[i:i+len(delta_a)]
    delta_w[i] = np.sum(x_temp*delta_a)
print(delta_w)

delta_x = np.zeros(len(X))

for i in range(len(X)):
    zero = np.zeros(len(delta_a)-1)
    w_padded = np.concatenate([zero,w,zero], axis =0)
    w_temp = w_padded[i:i+len(delta_a)]
    delta_x[i] = np.sum(w_temp*delta_a[::-1])
print(delta_x)

x = np.array([[1,2,3,4],[2,3,4,5]])
w = np.array([[[1,1,2],[2,1,1]], [[2,1,1],[1,1,1]], [[1,1,1],[1,1,1]]])
b = np.array([1,2,3])

a = np.zeros([3, output_size_calculation(4,3,0,1)])

for och in range(w.shape[0]):
    for ch in range(w.shape[1]):
        for m in range(a.shape[1]):
            a[och,m] += np.sum(x[ch, m:m+w.shape[2]]* w[och,ch,:])

a += b[:,None]
print("print forward prop:", a)

delta_a = np.array([[9,11], [32,35],[52,56]])

print("delta_a:\n", delta_a)
print("delta_a.shape:\n", delta_a.shape)

delta_b = np.sum(delta_a, axis = 1)
print("delta_b: ", delta_b)

delta_w = np.zeros([3,2,3])

for och in range(w.shape[0]):
    for ich in range(w.shape[1]):
        for fs in range(w.shape[2]):
            for m in range(2):
                delta_w[och,ich,fs] += (x[ich, fs+m]*delta_a[och,m])

print("delta_w:\n", delta_w)

delta_x = np.zeros([2,4])

for och in range(w.shape[0]):
    for ich in range(w.shape[1]):
        for fs in range(w.shape[2]):
            for m in range(2):
                delta_x[ich,fs+m] += w[och, ich,fs] * delta_a[och,m]

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train data shape: ", X_train.shape) # (60000, 28, 28)
print("X_test data shape: ", X_test.shape) # (10000, 28, 28)

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
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

NN = {0: FC(15640, 400, HeInitializer(), AdaGrad(0.01), Tanh()),
    1: FC(400, 200, HeInitializer(), AdaGrad(0.01), Tanh()),
    2: FC(200, 10, SimpleInitializer(0.01), AdaGrad(0.01), Softmax()),}

CNN = {0: SimpleConv1d(output_channel=20, input_channel=1, filter_size=3, padding=0, stride=1, 
            initializer=SimpleInitializerConv1d(0.01), optimizer=SGD(0.01), activation=Sigmoid()),}

cnn1d = Scratch1dCNNClassifier(NN=NN, CNN=CNN, n_epoch=10, n_batch=100, verbose = True)
cnn1d.fit(X_train[0:1000], y_train_one_hot[0:1000])

y_pred = cnn1d.predict(X_val[0:500])
acc = accuracy_score(y_val[0:500], y_pred)
print("Accuracy:", acc)
