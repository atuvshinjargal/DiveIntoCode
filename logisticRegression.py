import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.colors
import pickle as pkl


########### 1 - 4 ###################

class ScratchLogisticRegression():
    """
    ロジスティック回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    no_bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      訓練データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証データに対する損失の記録

    """

    def __init__(self, num_iter, lr, no_bias, verbose, lam=0.1):
        # ハイパーパラメータを属性として記録
        self.iter = num_iter
        self.lr = lr
        self.bias = no_bias
        self.verbose = verbose
        self.theta = None
        self.lam = lam
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        ロジスティック回帰を学習する。検証データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            訓練データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証データの正解値
        """

        if self.bias == True:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))
            if X_val is not None:
                bias = np.ones((X_val.shape[0], 1))
                X_val = np.hstack((bias, X_val))

        self.theta = np.zeros(X.shape[1])
        self.theta = self.theta.reshape(X.shape[1], 1)

        for i in range(self.iter):
            pred = self._logistic_hypothesis(X)       
            self._gradient_descent(X, y)
            loss = self._loss_func(pred, y)
            self.loss = np.append(self.loss, loss)     
            if X_val is not None:
                pred_val = self._logistic_hypothesis(X_val)
                loss_val = self._loss_func(pred_val, y_val)
                self.val_loss = np.append(self.val_loss, loss_val)

            if self.verbose == True:
                print('{}回目の学習の損失は{}'.format(i,loss))


    def predict(self, X):
        """
        ロジスティック回帰を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """

        if self.bias == True:
            a = np.ones(X.shape[0]).reshape(X.shape[0], 1)
            X = np.hstack([a, X])
        return  np.where(self._logistic_hypothesis(X) >= 0.5,1,0)

    def predict_proba(self, X):
        """
        ロジスティック回帰を使い確率を推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            ロジスティック回帰による推定結果
        """

        if self.bias == True:
            a = np.ones(X.shape[0]).reshape(X.shape[0], 1)
            X = np.hstack([a, X])
        pred = self._logistic_hypothesis(X)
        return pred

    def _sigmoid(self,y):
        """
        Calculate sigmoid function 

        Parameters
        ----------
        y : 次の形のndarray, shape (n_samples, )
            訓練データの正解値

        Returns
        -------
            次の形のndarray, shape (n_samples, )
            
        """
        sigmoid = 1 / (1 + np.exp(-y))
        return sigmoid

    def _logistic_hypothesis(self, X):
        """
        The hypothetical function of logistic regression

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            
        """
        pred = X @ self.theta
        pred = self._sigmoid(pred)
        return pred

    def _gradient_descent(self, X, y):
        m = X.shape[0]
        n = X.shape[1]
        pred = self._logistic_hypothesis(X)
        y = y.reshape((len(y),1))
        gradient = X.T @ (pred - y)
        self.theta = self.theta - (self.lr/m) * gradient


    def _loss_func(self, pred, y):
        error = 0
        for i in range(y.shape[0]):
            error += -np.sum(y[i] *  np.log(pred[i])+(1-y[i]) *  np.log(1-pred[i]))
        loss = error / (y.shape[0])
        loss = loss + np.sum(self.theta**2)*self.lam/(2 * y.shape[0])
        return loss
    
def decision_region(X,y,slr):
    mesh_f0, mesh_f1  = np.meshgrid(
        np.arange(np.min(X[:,0]), np.max(X[:,0]), 0.01), 
        np.arange(np.min(X[:,1]), np.max(X[:,1]), 0.01)
    )
    mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]
    print("mesh shape:{}".format(mesh.shape))
    y_pred = slr.predict(mesh).reshape(mesh_f0.shape)
    plt.title('decision region')
    plt.xlabel('feature0')
    plt.ylabel('feature1')
    plt.contourf(mesh_f0, mesh_f1, y_pred,cmap=matplotlib.colors.ListedColormap(['pink', 'skyblue']))
    plt.contour(mesh_f0, mesh_f1, y_pred,colors='red')
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1],label='0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1],label='1')
    plt.legend()
    plt.show()

###########  5 ###################
# Learning and estimation

iris = load_iris()
X = iris.data[:100,:]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


slr = ScratchLogisticRegression(num_iter=100, lr=0.1, no_bias=True, verbose=True, lam=0.1)
slr.fit(X_train, y_train, X_test, y_test)
pred = slr.predict(X_test)
print(pred.shape)
acc = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)

print("Scratch logistic regression: Accuracy - {}, Precision - {}, Recall - {}".format(acc, precision, recall))

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

pred = log_reg.predict(X_test)
acc = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
print("Scratch logistic regression: Accuracy - {}, Precision - {}, Recall - {}".format(acc, precision, recall))

########### 6 ###################
# Plot of learning curve

plt.plot(slr.loss, label = 'Training loss')
plt.plot(slr.val_loss, label = 'Validation loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Learning curve')
plt.show()

########### 7 ###################
# Visualization of decision area
X = iris.data[:100,:2]
y = iris.target[:100]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
slr = ScratchLogisticRegression(num_iter=1000, lr=0.01, no_bias=True,verbose=True,lam = 0.1)
slr.fit(X_train, y_train,X_test,y_test)

decision_region(X_test, y_test, slr)

########### 8 ###################
# (Advance task) Saving weights

# save the model to disk
filename = 'result.sav'
pkl.dump(slr, open(filename, 'wb'))

load_slr = pkl.load(open(filename,'rb'))
result = load_slr.predict(X_test)
print(result)