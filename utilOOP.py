from matplotlib import artist
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import warnings

data = load_iris()
X = data.data[:10]
scaler = StandardScaler()
scaler.fit(X)
print("平均 :", scaler.mean_)
print("分散 :", scaler.var_)
X_std = scaler.transform(X)
print('x_std: {}'.format(X_std))


# List of classes used so far

print('\x1b[6;30;42m' + "From Matplotlib we often use 'subplots'" + '\x1b[0m')
print('\x1b[6;30;42m' + "From Pandas we often use 'Dataframe'" + '\x1b[0m')
print('\x1b[6;30;42m' + "From Sklearn we often use 'classifier algorithms', i.e. support vector machine(SVM). " + '\x1b[0m')

# List of classes used so far
'''
We usually used Dataframe from pandas:
    iris dataset, household dataset and  - info(), df, data, X and y instances. head(), describe() ...  
We usually used Subplots from matplotlib:
    many assignments - ax instances. violinplot(), title(), plot(), scatter(), box(), are methods
We usually used ndarray from numpy: a_ndarray instances - sum(), shape(),  ...
We usually used LinearRegression from sklearn
    lr instance - score(), set_params(), get_params(), fit(), predict(), 
'''

# Create a standardized class by scratch

class ScratchStandardScaler():
    """
    標準化のためのクラス

    Attributes
    ----------
    mean_ : 次の形のndarray, shape(n_features,)
        平均
    var_ : 次の形のndarray, shape(n_features,)
        分散
    """

    def fit(self, X):
        """
        標準化のために平均と標準偏差を計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            訓練データ
        """

        self.mean_ = np.mean(X, axis= 0)
        self.var_ = np.var(X, axis = 0)

        pass

    def transform(self, X):
        """
        fitで求めた値を使い標準化を行う。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            特徴量

        Returns
        ----------
        X_scaled : 次の形のndarray, shape (n_samples, n_features)
            標準化された特徴量
        """
        X_scaled = np.subtract(X, self.mean_)
        X_scaled = np.divide(X_scaled,np.sqrt(self.var_))
        #X_scaled = 1
        pass
        return X_scaled



scratch_scaler = ScratchStandardScaler()
scratch_scaler.fit(X)
print("平均 : {}".format(scratch_scaler.mean_))
print("分散 : {}".format(scratch_scaler.var_))
X_std = scratch_scaler.transform(X)
print(X_std)

#### Special method ######## 

class arithmetic():
    """
    Calculate basic four arithmetic operation including add, substrate, multiply and divide

    Parameters
    ----------
    value : float or int
        初期値

    Attributes
    ----------
    value : float or int
        計算結果
    """
    def __init__(self, value):
        self.value = value
        print("初期値{}が設定されました".format(self.value))
        if not isinstance(value, (int, float)):
            raise Exception("Sorry, value must be int or float")
    def add(self, X):
        """
        受け取った引数をself.valueに加える
        
        Parameters
        ----------
        X : float or int
            received argument
        """
        if (self.check_type(X)):
            self.value += X
    def sub(self, X):
        """
        Substrate the received argument to self.value

        Parameters
        ----------
        X : float or int
            received argument
        """
        if (self.check_type(X)):
            self.value -= X
    def mul(self, X):
        """
        Multiply the received argument by self.value

        Parameters
        ----------
        X : float or int
            received argument
        """
        if (self.check_type(X)):
            self.value *= X
    def div(self, X):
        """
        Divide the received argument by self.value

        Parameters
        ----------
        X : float or int
            received argument
        """
        if (self.check_type(X)):
            self.value /= X
    def check_type(self, X):
        """
        Check input variable type. If it is not int or float then raise error.

        Parameters
        ----------------------
        X: any type
            received argument
        """
        if not isinstance(X, (int, float)):
            try:
                raise ValueError("Sorry, value must be int or float.")
            except ValueError as ex:
                print(ex.args)
            return  False
        return True

    

val = arithmetic(5)
print("value : {}".format(val.value))
val.add(3)
print("value : {}".format(val.value))
val.sub(2)
print("value : {}".format(val.value))
val.mul(5)
print("value : {}".format(val.value))
val.div(10)
print("value : {}".format(val.value))

check = arithmetic(10)
check.add('b')