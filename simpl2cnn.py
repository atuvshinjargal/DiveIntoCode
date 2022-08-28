import numpy as np
import random
from sklearn.metrics import accuracy_score
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

(X_train, t_train), (X_test, t_test) = mnist.load_data()
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
X_train /= 255
X_test /= 255
enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
t_train_one_hot = enc.fit_transform(t_train[:, np.newaxis])
t_test_one_hot = enc.fit_transform(t_test[:,  np.newaxis])
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

class Conv2d:
    
    def __init__(self, initializer, out_chanel, in_chanel, height, width, optimizer):
        init = initializer
        self.w = init.W(out_chanel, in_chanel, height, width)
        self.b = init.B(out_chanel)
        self.optimizer = optimizer
    
    def forward(self, X):
        self.sample_size, self.in_chanel, self.x_height, self.x_width = X.shape
        self.out_chanel, self.inchanel, self.w_height, self.w_width = self.w.shape
        self.XB = X
        
        
        A  = np.zeros([self.sample_size,  self.out_chanel, self.x_height -2, self.x_width -2])
        for n in range(self.sample_size):
            for outchan in range(self.out_chanel):
                for inchan in range(self.in_chanel):
                    for i in range(self.x_height-2):
                        for j in range(self.x_width-2):
                            sig = 0
                            for s in range(self.w_height):
                                for t in range(self.w_width):
                                    sig += X[n, inchan, i+s, j+t] * self.w[outchan, inchan, s, t]
                        A[n, outchan, i, j] += sig + self.b[outchan]
        return A
    
    def backward(self, dA):
        n_out_h , n_out_w = N_out(self.x_height, self.x_width, 0, self.w_height, self.w_width, 1)
        self.lb = dA.sum(axis=(0, 2, 3))
        
        self.lw = np.zeros_like(self.w)
        for n in range(self.sample_size):
            for m in range(self.out_chanel):
                for k in range(self.in_chanel):
                    for s in range(self.w_height):
                        for t in range(self.w_width):
                            for i in range(self.w_height-1):
                                for j in range(self.w_width-1):
                                    self.lw[m, k, s, t] += dA[n, m, i, j] * self.XB[n, k, i+s, j+t]
        
        
        dZ = np.zeros_like(self.XB)
        for n in range(self.sample_size):
            for m in range(self.out_chanel):
                for k in range(self.inchanel):
                    for i in range(self.x_height):
                        for j in range(self.x_width):
                            sig = 0
                            for s in range(self.w_height):
                                for t in range(self.w_width):
                                    if i-s<0 or i-s>n_out_h-1 or j-t < 0 or j-t>n_out_w-1:
                                        pass
                                    else:
                                        sig += dA[n, m, i-s, j-t] * self.w[m, k, s, t]
                            dZ[n, k, i, j] += sig
        
        
        
        self = self.optimizer.update(self)
        return dZ

def N_out(Nh_in, Nw_in, P, Fh, Fw, S):
    Nh_out = ((Nh_in  + 2*P - Fh) / S) + 1
    Nw_out = ((Nw_in + 2*P-Fw) / S) + 1
    return int(Nh_out), int(Nw_out)

class MaxPool2D:
    """
    pooling層のクラス
    2*2の正方行列内の最大値をリストに格納
    リストをreshapeすることで行列として返す
    
    2*2の正方行列内の最大値のインデックスを縦横それぞれ別のリストに格納
    要素が0の行列を作り、backwardで帰ってきた値を最大値のインデックスがあった場所へ代入
    """
    
    def forward(self,A):
        self.AB = A
        pooling_list = []
        self.h_list = []
        self.w_list = []
        for n in range(A.shape[0]):
            for cha in range(A.shape[1]):
                for h in range(0, 26, 2):
                    for w in range(0, 26, 2):
                        pooling_list = np.append(pooling_list, np.max(A[n, cha, h:h+2, w:w+2]))
                        self.h_list.append(np.unravel_index(np.argmax(A[n, cha, h:h+2, w:w+2]), A[n, cha, h:h+2, w:w+2].shape)[0])
                        self.w_list.append(np.unravel_index(np.argmax(A[n, cha, h:h+2, w:w+2]), A[n, cha, h:h+2, w:w+2].shape)[1])
                        
        pooing = pooling_list.reshape(A.shape[0], A.shape[1], 13, 13)
        return pooing
    
    def backward(self, dA):
        d = np.zeros_like(self.AB)
        i = 0
        for n in range(dA.shape[0]):
            for cha in range(dA.shape[1]):
                for h in range(0, 26, 2):
                    for w in range(0, 26, 2):
                        d[n, cha, h:h+2, w:w+2][self.h_list[i]][self.w_list[i]] = dA.reshape(-1)[i]
                        i+=1
        return d

class Flatten:
    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def bakcward(self, dA):
        return dA.reshape(self.X_shape)

class FC:
    """
    ノード数n_nodes1からn_nodes2への全結合層
    Parameters
    ----------
    n_nodes1 : int
      前の層のノード数
    n_nodes2 : int
      後の層のノード数
    initializer : 初期化方法のインスタンス
    optimizer : 最適化手法のインスタンス
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.optimizer = optimizer
        # 初期化
        # initializerのメソッドを使い、self.Wとself.Bを初期化する
        init = initializer
        self.n_nodes1 = n_nodes1
        self.w = init.W(n_nodes1, n_nodes2)
        self.b = init.B(n_nodes2)
    

    
    def forward(self, X):
        """
        フォワード
        Parameters
        ----------
        X : 次の形のndarray, shape (batch_size, n_nodes1)
            入力
        Returns
        ----------
        A : 次の形のndarray, shape (batch_size, n_nodes2)
            出力
        """
        self.z = X
        self.a = X@self.w + self.b
        
        return self.a

    
    def backward(self, dA):
        """
        バックワード
        Parameters
        ----------
        dA : 次の形のndarray, shape (batch_size, n_nodes2)
            後ろから流れてきた勾配
        Returns
        ----------
        dZ : 次の形のndarray, shape (batch_size, n_nodes1)
            前に流す勾配
        """
        dZ = dA @ self.w.T
        self.lw = self.z.T @ dA
        self.lb = np.sum(dA, axis=0)
        
        
        # 更新
        self = self.optimizer.update(self)
        return dZ
    
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
        重みの初期化
        Parameters
        ----------
        n_nodes1 : int
          前の層のノード数
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        W :
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
        B :
        """
        B  = self.sigma * np.random.randn(n_nodes2)
        return B
    
class SimpleInitializer_cnn:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def W(self, out_chanel, in_chanel, height, width):

        W = self.sigma * np.random.randn(out_chanel, in_chanel, height, width)
        
        return W
    
    def B(self, out_chanel):
        """
        バイアスの初期化
        Parameters
        ----------
        n_nodes2 : int
          後の層のノード数

        Returns
        ----------
        B :
        """
        B  = self.sigma * np.random.randn(out_chanel)
        return B
    
class SGD:

    def __init__(self, lr):
        self.lr = lr
    def update(self, layer):
        
        layer.w = layer.w -  self.lr * layer.lw
        layer.b = layer.b - self.lr*layer.lb
        
        return layer
    
    
class Softmax:
    def forward(self, A):
        exp_a = np.exp(A)
        softmax_result = np.empty((A.shape[0], A.shape[1]))
        exp_sum = np.sum(exp_a, axis=1)
        for i in range(A.shape[0]):
            softmax_result[i] = exp_a[i] / exp_sum[i]
            
        return softmax_result
    
    def backward(self, Z, Y):
        
        L_A = Z - Y
        self.cross_entropy = -np.average(np.sum(Y*np.log(Z), axis=1))
        
        
        return L_A

class Relu:
    def forward(self, X):
        self.A = X
        return np.maximum(0, X)
    
    def backward(self, Z):
        
        return Z * np.maximum(np.sign(self.A), 0)

X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
X_train_min = X_train[:3000]
t_train_min = t_train_one_hot[:3000]
X_test_min = X_test[:500]
t_test_min = t_test_one_hot[:500]

conv2d = Conv2d(SimpleInitializer_cnn(0.01), 3, 1, 3, 3, SGD(0.1))
maxpool = MaxPool2D()
relu = Relu()
fc1 = FC(507, 10, SimpleInitializer(0.01), SGD(0.1))
softmax = Softmax()
flat = Flatten()

A = conv2d.forward(X_train_min)
A_1 = relu.forward(A)
A_2 = maxpool.forward(A_1)
A_3 = flat.forward(A_2)
A_4 = fc1.forward(A_3)
A_5 = softmax.forward(A_4)
dA_1 = softmax.backward(A_5, t_train_min)
dA_2 = fc1.backward(dA_1)
dA_3 = flat.bakcward(dA_2)
dA_4 = maxpool.backward(dA_3)
dA_5 = relu.backward(dA_4)
dA_6 = conv2d.backward(dA_5)

test_A = conv2d.forward(X_test_min)
test_A_2 = relu.forward(test_A)
test_A_3 = maxpool.forward(test_A_2)
test_A_4 = flat.forward(test_A_3)
test_A_5 = fc1.forward(test_A_4)
test_A_6 = softmax.forward(test_A_5)
y = np.argmax(test_A_6 , axis=1)
print('accuracy', accuracy_score(t_test[:500], y))