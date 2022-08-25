from pickletools import optimize
from re import A
from turtle import forward
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

class SimpleInitializer:
    """
    ガウス分布によるシンプルな初期化
    Parameters
    ----------
    sigma : float
      ガウス分布の標準偏差
    """
    def __init__(self, sigma):
        self.sigma = sigma

    def W(self, n_nodes1, n_nodes2):
        """
        Weight initialization
        Parameters
        ----------
        n_nodes1 : int
          Number of nodes in the previous layer
        n_nodes2 : int
          Number of nodes in the later layer

        Returns
        ----------
        W : ndarray shape with (n_nodes1, n_nodes2)
          weights of hidden layer
        """
        W = self.sigma * np.random.randn(n_nodes1, n_nodes2)
        return W

    def B(self, n_nodes2):
        """
        バイアスの初期化
        Parameters
        ----------
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        B : ndarray shape with (n_nodes2, 1)
          bias of hidden layer
        """
        B = np.zeros(n_nodes2)
        return B

class SGD:
    """
    Stochastic gradient descent
    Parameters
    ----------
    lr : learning rate
    """
    def __init__(self, lr):
        self.lr = lr

    def update(self, layer):
        """
        Update weights and biases for a layer
        Parameters
        ----------
        layer : Instance of the layer before update
        """

        #layer.w_x -= self.lr*layer.dw_x
        #layer.b -= self.lr*layer.db
        #layer.w_h -= self.lr*layer.dw_h

        layer.w_x[...] = layer.w_x - self.lr * np.dot(layer.X.T, layer.dA) / len(layer.dA)
        layer.b[...] = layer.b - self.lr * np.mean(layer.dA)
        layer.w_h[...] = layer.w_h[...] - self.lr * np.dot(layer.h_t.T, layer.dA) / len(layer.dA)

        return layer

class tanh():
  """
  Activation function - tangent
  """
  def __init__(self):
      pass
  
  def forward(self, A):
    self.A = A
    self.Z = np.tanh(self.A)

    return self.Z

  def backward(self, dZ):
    ret = dZ * (1-self.Z**2)
    return ret

class sigmoid():
  """
  Activation function - sigmoid
  """
  def __init__(self):
      pass
  
  def forward(self, A):
    self.A = A
    self.Z = 1/(1 + np.exp(-self.A))

    return self.Z

  def backward(self, dZ):
    ret = dZ * (1-self.Z)*self.Z
    return ret
    
class softmax():
  """
  Activation function - softmax
  """
  def __init__(self):
      pass
  
  def forward(self, A):
    #print("A:", A)
    #print("temp:", np.exp(A - np.max(A)))
    #print("forward temp:", np.sum(np.exp(A-np.max(A))))
    temp = np.exp(A - np.max(A))/np.sum(np.exp(A-np.max(A)), axis = 1, keepdims= True)
    return temp

  def backward(self, dZ):
    return dZ
  
class ReLU():
  """
  Activation function - relu
  """
  def __init__(self):
      pass
  
  def forward(self, A):
    self.A = A
    temp = np.maximum(self.A, 0)
    return temp

  def backward(self, dZ):
    ret = np.where(self.A>0, dZ, 0)
    return ret


  """
  stochastic gradient descent method

  Parameters
  -----------
  lr : learning rate
  """
  def __init__(self, lr):
      self.lr = lr
      self.hW = 0
      self.hB = 0
  
  def update(self,layer):
    """
    Updating the weights and biases of a layer
    Parameters
    -----------
    layer : Instance of the layer before the update
    """
    self.hW += layer.dW * layer.dW
    self.hB = layer.dB * layer.dB

    layer.W -= self.lr * layer.dW/(np.sqrt(self.hW) + 1e-7)
    layer.B -=self.lr * layer.dB/(np.sqrt(self.hB) + 1e-7)

    return layer

class GetMiniBatch:
    """
    Iterator to get a mini-batch
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
      training data
    y : ndarray, shape (n_samples, 1)
      Label of training data
    batch_size : int
      batch size
    seed : int
      NumPy random seed
    """
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

##################### 1 ###################
# SimpleRNN forward propagation implementation

class SimpleRNN:
    """
    Simple recurrent neural network
    Parameters
    ----------
    n_nodes1 : int
      Number of nodes in the previous layer
    n_nodes2 : int
      Number of nodes in the next layer
    initializer : Instance of initialization method
    optimizer : Instance of optimization method
    activation : activation function 
    """
    def __init__(self, w_x, b, w_h, n_nodes1, n_nodes2, initializer, optimizer, activation):
        self.optimizer = optimizer
        self.initializer = initializer
        self.n_nodes1 = n_nodes1
        self.n_nodes2 = n_nodes2
        self.activation = activation
        # Initialization
        # Initialize self.W and self.B using the #initialr method

        self.WX = initializer.W(n_nodes1, n_nodes2)
        self.Wh = initializer.W(n_nodes2, n_nodes2)
        self.B = initializer.B(1)

        self.w_x = w_x
        self.b = b
        self.w_h = w_h
        
    def forward(self, x):
        """
        forward propagation

        Parameters
        ----------
        x : ndarray, shape (batch_size, n_nodes1)
            input
        Returns
        ----------
        h : ndarray, shape (batch_size, n_nodes2)
            output
        """        
        self.x_in = x
        b_size, n_sequences, n_features = self.x_in.shape 
        h_t = np.zeros((b_size, self.n_nodes2))
        A = np.empty((0, b_size, self.n_nodes2))
        for i in range(n_sequences):
            h_t = np.dot(self.x_in[:, i, :].reshape(b_size, n_features), self.w_x) + np.dot(h_t, self.w_h) + self.b
            h_t = self.activation.forward(h_t)
            A = np.vstack((A, h_t[np.newaxis,:])) #shape (sequence, batch, n_node)
        A = A.transpose(1, 0, 2)
        self.A = A
        print("A:", A)
        return h_t
        
    def backward(self, dA):
        """
        backward propagation
        Parameters
        ----------
        dA : ndarray, shape (batch_size, n_nodes2)
            Gradient descent from behind
        Returns
        ----------
        dZ : ndarray, shape (batch_size, n_nodes1)
            Gradient descent forward
        """
        b_size, n_sequences, n_nodes = self.A.shape
        dZ = np.zeros((n_sequences, b_size, self.n_nodes1))
        for i in reversed(range(n_sequences)):
            dA = self.activation.backward(dA)
            dA = dA * (1 - self.A[:, i, :]**2) #shape (batch,n_nodes)
            self.dA = dA
            print("dA shape:", self.dA.shape)
            self.X = self.x_in[:, i, :]
            self.h_t = self.A[:, i, :]
            self = self.optimizer.update(self)
            dA = np.dot(dA, self.w_h.T) #shape(batch, n_nodes)
            dZ[i, :, :] = np.dot(dA, self.w_x.T) #shape (batch, n_features)
            
        dZ = dZ.transpose(1,0,2)
        return dZ

#################### 2 ####################
# Experiment of forward propagation with a small array

x = np.array([[[1, 2], [2, 3], [3, 4]]])/100 # (batch_size, n_sequences, n_features)
w_x = np.array([[1, 3, 5, 7], [3, 5, 7, 8]])/100 # (n_features, n_nodes)
w_h = np.array([[1, 3, 5, 7], [2, 4, 6, 8], [3, 5, 7, 8], [4, 6, 8, 10]])/100 # (n_nodes, n_nodes)
batch_size = x.shape[0] # 1
n_sequences = x.shape[1] # 3
n_features = x.shape[2] # 2
n_nodes = w_x.shape[1] # 4
h = np.zeros((batch_size, n_nodes)) # (batch_size, n_nodes)
b = np.array([1, 1, 1, 1]) # (n_nodes,)

#w_x, b, w_h, n_nodes1, n_nodes2, initializer, optimizer, activation
rnn = SimpleRNN(w_x, b, w_h, 2, 4, SimpleInitializer(0.01), SGD(lr=0.1), tanh())
h = rnn.forward(x)
print("h:", h)



