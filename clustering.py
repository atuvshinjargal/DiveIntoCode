from cProfile import label
from enum import auto, unique
from pyexpat import features
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from matplotlib import cm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

################# 1 #########################
# Implementation of K-means cluster

class ScratchKMeans():
    """
    Implementation of K-means cluster

    Parameters
    ----------
    n_clusters : int
      number of class
    n_init : int
      How many times to change the initial value of the center point to calculate
    max_iter : int
      How many iterations can be done in one calculation
    tol : float
      Margin of error between the center point and the center of gravity, which is the reference for ending the iteration
    verbose : bool
      True to output the learning process
    """
    def __init__(self, n_clusters, n_init, max_iter, tol= 1e-5, verbose=False):
        # Record hyperparameters as attributes
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.record_myu = None
        self.record_cluster = None


    def fit(self, X):
        """
        Calculate clustering by K-means

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features of training data
        """

        for i in range(self.n_init):
          self.myu = X[np.random.choice(len(X),size=self.n_clusters,replace=False)]
          for j in range(self.max_iter):
            self._update_cluster(X)
            old_myu = self.myu
            self._update_myu(X)

            if np.sum(old_myu - self.myu) < self.tol:
              self.calc_sse(X)
              break
            
            self.calc_sse(X)

        #fig = plt.subplots(figsize = (10,8))
        #plt.rcParams['font.size'] = 20
        #for i in range(self.n_clusters):
        #    plt.scatter(X[X_cluster[:,-1]==i,0],X[X_cluster[:,-1]==i,1], s=80)
        #plt.scatter(self.myu[:,0], self.myu[:,1], s = 100)
        #plt.show()

    def calc_sse(self,X):
        """
        Calculate SSE(Sum of Squared Error)
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features + 1)
            Features of training data
        """
        self.best_sse = 10**20
        sse = 0

        diff = (X-self.myu[:,None]).reshape(-1,2)
        self.dist = np.linalg.norm(diff, ord= 2, axis=1).reshape(self.n_clusters, -1).T
        sse = np.sum(np.min(self.dist, axis=1)**2)
        
        if self.best_sse > sse:
            self.best_sse = sse
            self.record_myu = self.myu
            self.record_cluster = self.n_clusters
            self._silhouette(X) if self.n_clusters >=2 else _
            

    def _update_cluster(self,X):
        """
        Assignment to the cluster

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features + 1)
            Features of training data
        """

        diff = (X - self.myu[:,None]).reshape(-1,2)
        self.dist = np.linalg.norm(diff, ord = 2, axis= 1).reshape(self.n_clusters,-1).T
        self.cluster = np.argmin(self.dist, axis=1)

    def _update_myu(self,X):
        """
        Moving the center point by update value of mu.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Features of training data
        """

        for j in range(self.n_clusters):
           self.myu[j] = np.mean(X[self.cluster == j],axis=0)
    
    def _silhouette(self, X):
      """
      silhouette diagram to help determine k.

      Parameters
      -------------------
      X : ndarray, shape (n_samples, n_features)
            Features of training data
      """
      ab = np.zeros([len(X),2])

      self.silhouette_vals = np.zeros(len(X))

      for i,j in enumerate(X):
        ab[i,0] = np.sum(np.linalg.norm(j-X[self.cluster == self.cluster[i]], ord=2,axis=1))/len(X[len(X[self.cluster == self.cluster[i]])]-1)
        ab[i,1] = np.mean(np.linalg.norm(j-X[self.cluster == self.dist[i].argsort()[1]], ord=2, axis=1))

        self.silhouette_vals[i] = (ab[i,1] - ab[i,0])/np.max(ab[i])
      
      self.silhouette_avg = np.mean(self.silhouette_vals)
      self.y_km = self.cluster
      self.cluster_labels = np.unique(self.cluster)


    def predict(self, X):
        """
        Calculate which cluster the entered data belongs to
        """
        diff = (X- self.record_myu[:,None]).reshape(-1,2)
        dist = np.linalg.norm(diff, ord=2, axis=1).reshape(self.record_cluster,-1).T
        return np.argmin(dist, axis = 1)



################# Problem 7 #########################
# Presumption
X, _ = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=0.5, shuffle=True, random_state=0)

print("X shape: {} and type: {}".format(X.shape, type(X)))


X_train, X_valid = train_test_split(X,train_size=0.8,random_state=None)

clf = ScratchKMeans(n_clusters=4,n_init=5,max_iter=100,verbose=True)
clf.fit(X_train)
pred = clf.predict(X_valid)
print(pred)

################# Problem 8 #########################
# Implementation of elbow method

elbow = {}
for k in range(1,8):
    model = ScratchKMeans(n_clusters=k,n_init=5,max_iter=100,verbose=False)
    model.fit(X_train)
    elbow[k] = model.best_sse
fig = plt.subplots(figsize=(12,8))
plt.rcParams["font.size"] = 16
plt.plot(list(elbow.keys()),list(elbow.values()),'rs--')
plt.title("Elbow method graph")
plt.xlabel('number of cluster')
plt.ylabel("SSE")
plt.show()

################# Problem 9 #########################
# (Advance task) Silhouette diagram
'''
plt.rcParams['font.size'] = 20

for k in range(2,8):
  model = ScratchKMeans(n_clusters=k, n_init=100, max_iter=1000, tol=1e-5)
  model.fit(X_train)

  if model.n_clusters >= 2:
    model._silhouette(X_train)
    
    y_ax_lower, y_ax_upper = 0, 0 
    yticks = []
    for i, c in enumerate(model.cluster_labels):
      c_silhouette_vals = model.silhouette_vals[model.y_km == c]
      c_silhouette_vals.sort()
      y_ax_upper += len(c_silhouette_vals)
      color = cm.jet(i / model.n_clusters)
      plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
      yticks.append((y_ax_lower + y_ax_upper) / 2)
      y_ax_lower += len(c_silhouette_vals)

    plt.axvline(model.silhouette_avg, color="red", linestyle="--")
    plt.yticks(yticks, model.cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()
'''
################# 10 #########################
# Selection of the number of clusters k

data = pd.read_csv('../data/Wholesale customers data.csv')
X = data[['Fresh', 'Milk', 'Grocery', 'Frozen','Detergents_Paper', 'Delicassen']].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

n = 2
pca = PCA(n_components=n)
pca = pca.fit(X)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)

plt.bar(range(1,n+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,n+1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.hlines(0.7, 0, n+1,  "blue", linestyles='dashed')
plt.legend(loc='best')
plt.grid()
plt.show()

pca_data = pca.transform(X)

elbow = {}
for k in range(1,8):
    model = ScratchKMeans(n_clusters=k,n_init=5,max_iter=100,verbose=False)
    model.fit(pca_data)
    elbow[k] = model.best_sse
fig = plt.subplots(figsize=(12,8))
plt.rcParams["font.size"] = 16
plt.plot(list(elbow.keys()),list(elbow.values()),'rs--')
plt.title("Elbow method graph for wholesale customer data")
plt.xlabel('number of cluster')
plt.ylabel("SSE")
plt.show()

print("Elbow method graph for wholesale customer data \n \
  k =3 is most optimal. Because the bending of graph is occured at k = 3.")


clf = ScratchKMeans(n_clusters=3,n_init=5,max_iter=100,verbose=False)
clf.fit(pca_data)
cluster = clf.y_km
data['cluster'] = cluster

pca_data_label = np.concatenate([pca_data, cluster.reshape(-1,1)], axis=1)

fig = plt.subplots(figsize=(12,8))
for i in range(clf.n_clusters):
  plt.scatter(pca_data_label[cluster==i,0], pca_data_label[cluster == i,1])
plt.show()

################# 11 #########################
# Comparison with known group

feature = 'Channel'
fig = plt.figure()
ax = fig.add_subplot()
ax = sns.distplot(data[feature])
plt.show()

feature = 'Region'
fig = plt.figure()
ax = fig.add_subplot()
ax = sns.distplot(data[feature])
plt.show()

feature = 'cluster'
fig = plt.figure()
ax = fig.add_subplot()
ax = sns.distplot(data[feature])
plt.show()

print('Channel and Region are separable \n \
  our identified cluster is much less separable.')
################# 12 #########################
# Useful information for wholesalers

print(data.groupby('cluster').agg(['mean', 'max', 'min']))

sns.pairplot(data, hue = 'cluster')

################# 13 #########################
# (Advance task) Survey of other methods
print('DBSCAN')
print("Advantages: - DBSCAN doesn't require to specify the number of cluster. \n \
    - Performs well with arbitrary shapes cluster. \n \
    - Robust to outliers and able to detect the outliers.")

print("Disadvantages: - Calculating distance of neighborhood is not easy and it requires domain knowledge.")

print('t-SNE')
print("Advantages: - Can work non linear data efficently whereas PCA is a linear algorithm. \n \
    - Preserves Local and Global Structure.")

print("Disadvantages: - Computationally complex \n \
  - Require hyperparameter tuning \n \
  - Noise patterns")

print('LLE')
print("Advantages: - Simple operation")

print("Disadvantages: - difficult to work large dataset")

################# 13 #########################
# (Advance task) Use of t-SNE and DBSCAN

X_sne = TSNE(n_components=2, learning_rate='auto').fit_transform(X)
print('X_sne.shape:{}'.format(X_sne.shape))

db_scan = DBSCAN(eps = 2, min_samples=2).fit(X_sne)
print(db_scan.labels_)

lab = np.unique(db_scan.labels_)

fig = plt.subplots(figsize=(12,8))
for i in lab:
  plt.scatter(pca_data_label[db_scan.labels_ ==i,0], pca_data_label[db_scan.labels_ == i,1])
plt.show()



