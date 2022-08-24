from cProfile import label
from traceback import format_tb
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class ScratchLinearRegression():
    """
    線形回帰のスクラッチ実装 - Scratch implementation of linear regression
    
    Parameters
    ----------
    num_iter : int
      イテレーション数 - Number of iterations
    lr : float
      学習率 - Learning rate
    no_bias : bool
      バイアス項を入れない場合はTrue - True if no bias term is included
    verbose : bool
      学習過程を出力する場合はTrue - True to output the learning process
    
    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,) - ndarray, shape (n_features,) of the following form
        パラメータ Parameters
    self.loss : 次の形のndarray, shape (self.iter,) - ndarray of the following form, shape (self.iter,)
      訓練データに対する損失の記録 - Recording losses for training data
    self.val_loss : 次の形のndarray, shape (self.iter,) - ndarray, shape (self.iter,) of the following form
      検証データに対する損失の記録 - Recording losses for validation data
    """
    
    def __init__(self, num_iter, lr, no_bias, verbose):
        # ハイパーパラメータを属性として記録 - Record hyperparameters as attributes
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.theta = None 
        # 損失を記録する配列を用意 - Prepare an array to record the loss
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        #self.theta = np.
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。
        Learn linear regression. If validation data is entered, the loss and accuracy for it are also calculated for each iteration.
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データの特徴量 - Features of training data
        y : 次の形のndarray, shape (n_samples, )
            訓練データの正解値 - Correct answer value of training data
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証データの特徴量 - Features of verification data
        y_val : 次の形のndarray, shape (n_samples, )
            検証データの正解値 - Correct value of verification data
        """
        if self.no_bias == True:
            bias = np.ones((X.shape[0], 1))
            X = np.hstack((bias, X))
            if X_val is not None:
                bias = np.ones((X_val.shape[0], 1))
                X_val = np.hstack((bias, X_val))
        self.theta = np.zeros(X.shape[1])
        self.theta = self.theta.reshape(X.shape[1], 1)
        for i in range(self.iter):
            pred = self._linear_hypothesis(X)
            self._gradient_descent(X, y)
            #print('in the linear hyp theta:{}'.format(self.theta))
            loss = self._loss_func(pred, y)
            #print('loss:{}'.format(loss))
            self.loss = np.append(self.loss, loss)         
            if X_val is not None:
                pred_val = self._linear_hypothesis(X_val)
                loss_val = self._loss_func(pred_val, y_val)
                self.val_loss = np.append(self.val_loss, loss_val)
            if self.verbose == True:
                print('The loss of the {} learning:{}'.format(i,loss))
        
    def predict(self, X):
        """
        線形回帰を使い推定する。 Estimate using linear regression.
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            サンプル sample
        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果 Estimated result by linear regression
        """
        if self.no_bias == True:
            bias = np.ones(X.shape[0]).reshape(X.shape[0], 1)
            X = np.hstack([bias, X])
        pred_y = self._linear_hypothesis(X)
        return pred_y

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する - Compute a linear hypothetical function

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
        訓練データ - Training data

        Returns
        -------
        次の形のndarray, shape (n_samples, 1)
        線形の仮定関数による推定結果 - Estimated result by linear hypothetical function

        """
        
        pred = np.dot(X,self.theta)
        return pred

    def _gradient_descent(self, X, y):
        """
        Perform gradient descent for n iteration, which involves making a prediction, computing the gradients and updating the weight and biases

        Parameters
        ----------------------
        X : 次の形のndarray, shape (n_samples, n_features)
        訓練データ - Training data

        y : 次の形のndarray, shape (n_samples, )
            訓練データの正解値 - Correct answer value of training data

        Returns
        ----------------------

        """
        m = X.shape[0]
        n = X.shape[1]
        pred = self._linear_hypothesis(X)
        gradient = X.T @ (pred - y)
        self.theta = self.theta - (self.lr/m) * gradient
  

        
    def _MSE(self, y_pred, y):
        """
        平均二乗誤差の計算 Calculation of mean square error

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples,)
        推定した値 - Estimated value
        y : 次の形のndarray, shape (n_samples,)
        正解値 - Correct answer value

        Returns
        ----------
        mse : numpy.float
        平均二乗誤差 Mean squared error
        """
        mse = ((y_pred - y) ** 2).sum() / y.shape[0]
        return mse
    
    def _loss_func(self,y_pred, y):
        loss = self._MSE(y_pred, y)/2
        return loss


train = pd.read_csv("../house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../house-prices-advanced-regression-techniques/test.csv")

X = train.loc[:, ['GrLivArea', 'YearBuilt']]
y = train.loc[:, ['SalePrice']]
X = X.values
y = y.values


############# 6 ############
#  Learning and estimation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=123)
#print(X_train)
#print(y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

slr = ScratchLinearRegression(num_iter=1000, lr=0.01, no_bias=True, verbose=False)
slr.fit(X_train, y_train, X_test, y_test)
pred = slr.predict(X_test)
#print("Prediction using ScrathLinearRegression:{}".format(pred))
score = mean_squared_error(y_test, pred)
print("MSE of ScratchLinearRegression:{}".format(score))


reg = LinearRegression().fit(X_train, y_train)
pred = reg.predict(X_test)
#print("Prediction using LinearRegression from sklearn:{}".format(pred))
score = mean_squared_error(y_test, pred)
print("MSE of sklearn Linear Regression:{}".format(score))

####### Problem 7 ############
# Plot of learning curve

plt.plot(slr.loss, label = 'Training loss')
plt.plot(slr.val_loss, label = 'Validation loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Learning curve')
plt.show()

####### 8 ###########
# (Advance task) Removal of bias term

slr = ScratchLinearRegression(num_iter=1000, lr=0.01, no_bias=False, verbose=False)
slr.fit(X_train, y_train, X_test, y_test)
pred = slr.predict(X_test)
#print("Prediction using ScrathLinearRegression:{}".format(pred))

###### 9 ###########
# (Advance task) Multidimensionalization of features

from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=2)

X = train.loc[:, ['GrLivArea', 'YearBuilt']]
y = train.loc[:, ['SalePrice']]

poly_transformer.fit(X)
X = poly_transformer.transform(X)
print('Polynomial Features shape: ', X.shape)

#X = X.values
y = y.values


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=123)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

slr = ScratchLinearRegression(num_iter=1000, lr=0.01, no_bias=True, verbose=False)
slr.fit(X_train, y_train, X_test, y_test)
pred = slr.predict(X_test)
#print("Prediction using ScrathLinearRegression with polinominal feature:{}".format(pred))
score = mean_squared_error(y_test, pred)
print("MSE of ScratchLinearRegression with polynominal feauters:{}".format(score))
print('When using polynominal feature the root mean square error is reduced')

######## 10 ###########
# (Advance task) Derivation of update formula

'''
The goal of the gradient descent algorithm is to minimize the given function (say cost function J(theta)). 
To achieve this goal, it performs two steps iteratively:

1. Compute the gradient (slope), the first order derivative of the function at that point. Derivative means we calculate difference of two points.
2. Make a step (move) in the direction opposite to the gradient, opposite direction of slope increase from the current point by alpha times the gradient at that point

Gradient descent starts with a random value of θ.

'''