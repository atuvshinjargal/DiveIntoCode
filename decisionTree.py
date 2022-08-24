import enum
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.colors

############ 1 - 4 ##############

class ScratchDecesionTreeClassifierDepth1():
    """
    Scratch implementation of a decision tree classifier of depth 1

    Parameters
    ----------
    verbose : bool
      True if you want to output the learning progress
    """

    def __init__(self, verbose=False):
        # Record hyperparameters as attributes
        self.verbose = verbose

    def fit(self, X, y):
        """
        Learn the decision tree classifier

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
           Features of training data
        y : ndarray, shape (n_samples,) of the following form
            Label value of training data
        """

        self.gain = 0
        for feature_index in range(X.shape[1]):
            self.thresholds = np.unique(X[:,feature_index])
            for threshold in self.thresholds:
                _gain = self._calc_ig(X, feature_index, y, threshold)
                if _gain > self.gain:
                    self.l_label = collections.Counter(y[X[:,feature_index] < threshold]).most_common()[0][0]
                    self.r_label = collections.Counter(y[X[:,feature_index] >= threshold]).most_common()[0][0]
                    self.feature = feature_index
                    self.threshold = threshold
                    self.gain = _gain
                    if self.verbose:
                        print('feature',self.feature,'threshold',self.threshold,'gain',self.gain)

    def predict(self, X):
        """
        Estimate the label using a decision tree classifier
        """
        prediction = np.where(X[:,self.feature] < self.threshold,self.l_label,self.r_label)
        return prediction
    
    def _calc_gini(self,y):
        """
        Calculate Gini Impurity

        Parameters
        --------------
        y : ndarray, shape (n_samples,) of the following form
            Label value of training data
        """
        labels, counts = np.unique(y, return_counts=True)
        _gini = 0
        for label in labels:
            _gini += (counts[labels==label]/np.sum(counts))**2
        gini = 1 - _gini
        return gini
    
    def _calc_ig(self,X,feature_index,y,threshold):
        """
        Calculate Information gain

        Parameters
        --------------
        X : nd array, shape (n_samples, feature_size) of the following form
            feature values of training data
        feature_index : int
            feature number
        y : nd array, shape (n_samples,) of the following form
            Label value of training data
        threshold : int
            threshold value
        """

        labels_l = y[X[:,feature_index] < threshold]
        labels_r = y[X[:,feature_index] >= threshold]
        gain_l = (len(labels_l)/(len(labels_l)+len(labels_r)))*self._calc_gini(labels_l)
        gain_r = (len(labels_r)/(len(labels_l)+len(labels_r)))*self._calc_gini(labels_r)
        gain = self._calc_gini(y) -gain_l -gain_r
        return gain

#########################################################################

class Node():
    """
    A class that composes and separates the nodes of a decision tree.

    Parameters:
    ---------------
    verbose : bool
        True if you want to output the learning progress
    max_depth = int
        Maximum training depth of decision tree
    """
    def __init__(self, max_depth = None, verbose = False):
        # Record hyperparameters as attributes
        self.verbose = verbose
        self.max_depth = max_depth
        self.feature = None

    def _separate(self, X, y, depth):
        """
        Split nodes using the CART method.
        Parameters:
        ---------------
        X : ndarray, shape (n_samples, n_features)
            Features of training data
        y : ndarray, shape (n_samples,) of the following form
            Label value of training data
        depth : int
            the depth of node
        """
        self.depth = depth
        features = X.shape[1]

        if len(np.unique(y))  == 1:
            self.label = np.unique(y)[0]
            return
        
        if self.depth == self.max_depth:
            self.label = statistics.mode(y)
            return
        
        self.gain = 0

        for feature_number in range(features):
            thresholds = np.unique(X[:,feature_number])
            for threshold_value in thresholds[1:]:
                labels_l = y[X[:,feature_number] < threshold_value]
                labels_r = y[X[:,feature_number] >= threshold_value]

                gain_l = (len(labels_l)/(len(labels_l)+len(labels_r)))*self._calc_gini(labels_l)
                gain_r = (len(labels_r)/(len(labels_l)+len(labels_r)))*self._calc_gini(labels_r)
                gain = self._calc_gini(y) -gain_l -gain_r

                if gain > self.gain:
                    self.feature = feature_number
                    self.threshold = threshold_value
                    self.gain = gain

        if self.verbose:
            print("depth", self.depth, 'feature',self.feature,'threshold',self.threshold,'gain',self.gain)
        
        l_X = X[X[:,self.feature] < self.threshold]
        l_y = y[X[:,self.feature] < self.threshold]
        self.left = Node(self.max_depth, self.verbose)
        self.left._separate(l_X, l_y, depth + 1)


        r_X = X[X[:,self.feature] >= self.threshold]
        r_y = y[X[:,self.feature] >= self.threshold]
        self.right = Node(self.max_depth, self.verbose)
        self.right._separate(r_X, r_y, depth + 1)
    
    def _calc_gini(self, y):
        """
        Calculate Gini Impurity

        Parameters
        --------------
        X : ndarray, shape (n_samples,) of the following form
            Features of training data
        """
        labels, counts = np.unique(y, return_counts=True)
        _gini = 0
        for label in labels:
            _gini += (counts[labels==label]/np.sum(counts))**2
        gini = 1 - _gini
        return gini
    
    def _predict(self, X):
        """
        Estimate label
        """
        if self.feature == None or self.depth == self.max_depth:
            return self.label
        else:
            if X[self.feature] < self.threshold:
                return self.left._predict(X)
            else:
                return self.right._predict(X)



class ScratchDecisionTreeClassifier():
    '''
    Scratch implementation of a decision tree classifier of depth n

    Parameters:
    ---------------
    verbose : bool
        True if you want to output the learning progress
    max_depth = int
        Maximum training depth of decision tree
    '''

    def __init__(self, max_depth = 1, verbose = False):
        # Record hyperparameters as attributes
        self.verbose = verbose
        self.max_depth = max_depth
    
    def fit(self, X, y):
        """
        Learn the decision tree classifier
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features of training data
        y : ndarray, shape (n_samples, )
            Label values of training data
        """

        self.depth = 0
        self.root = Node(self.max_depth, self.verbose)
        self.root._separate(X, y, self.depth)

    def predict(self, X):
        """"
        Estimate labels using a decision tree classifier
        """
        y_pred = np.zeros(len(X))
        
        for i, x in enumerate(X):
            y_pred[i] = self.root._predict(x)
        return y_pred


#########################################################################


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
############ 5 ##############


iris = load_iris()
X = iris.data[:100,:]
y = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = ScratchDecesionTreeClassifierDepth1(verbose = True)
clf.fit(X_train,y_train)

prediction = clf.predict(X_test)

acc = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, average='micro')
recall = recall_score(y_test, prediction, average='micro')

print('Scratch Desicion tree: Accuracy', acc, 'Precision ', precision, 'Recall ',recall)

dtc = DecisionTreeClassifier(max_depth=1)
dtc.fit(X_train, y_train)

pred = dtc.predict(X_test)
acc = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='micro')
recall = recall_score(y_test, pred, average='micro')

print('Depth = 1 Sklearn Desicion tree: Accuracy', acc, 'Precision ', precision, 'Recall ',recall)



############ 6 ##############
# Visualization of decision area

X = iris.data[:100,:2]
y = iris.target[:100]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
clf = ScratchDecesionTreeClassifierDepth1(verbose = True)
clf.fit(X_train,y_train)
decision_region(X_test, y_test, clf)

############ 7 ##############
# Creation of a decision tree classifier class with a depth of 2
clf2 = ScratchDecisionTreeClassifier(max_depth= 2, verbose = True)
clf2.fit(X_train, y_train)

pred = clf2.predict(X_test)
print(pred)
print(y_test)
acc = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='micro')
recall = recall_score(y_test, pred, average='micro')
print('Depth = 2 Scratch Desicion tree: Accuracy ',acc, 'Precision ', precision,'Recall', recall)

############ 8 ##############
#Creation of decision tree classifier class with unlimited depth

clf = ScratchDecisionTreeClassifier(max_depth= None, verbose = True)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(pred)
print(y_test)
acc = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred, average='micro')
recall = recall_score(y_test, pred, average='micro')
print('Depth = None Scratch Desicion tree: Accuracy', acc, 'Precision ', precision, 'Recall ',recall)