import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
tf.compat.v1.disable_eager_execution()
from keras.datasets import mnist


################ user defined classes #################
class SampleIterator():
    def __init__(self):
        self.X = [1,2,3,4,5]
        self.counter = 0
        self.stop = len(self.X)
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter >= self.stop:
            raise StopIteration()
        x = self.X[self.counter]
        self.counter += 1
        return x

class GetMiniBatch:
    def __init__(self, X, y, batch_size = 10, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self.X = X[shuffle_index]
        self.y = y[shuffle_index]
        self._stop = np.ceil(X.shape[0]/self.batch_size).astype(np.int)
    def __len__(self):
        return self._stop
    def __getitem__(self,item):
        p0 = item*self.batch_size
        p1 = item*self.batch_size + self.batch_size
        return self.X[p0:p1], self.y[p0:p1]        
    def __iter__(self):
        self._counter = 0
        return self
    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter*self.batch_size
        p1 = self._counter*self.batch_size + self.batch_size
        self._counter += 1
        return self.X[p0:p1], self.y[p0:p1]

############# dataset preparation ####################
df = pd.read_csv("Iris.csv")
df = df[(df["Species"] == "Iris-versicolor") | (df["Species"] == "Iris-virginica")]
y = df["Species"]
X = df.loc[:, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
X = np.array(X)
y = np.array(y)
y[y == "Iris-versicolor"] = 0
y[y == "Iris-virginica"] = 1
y = y.astype(np.int64)[:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

############################### 
sample_iter = SampleIterator()
for x in sample_iter:
    print(x)

###############################
learning_rate = 0.001
batch_size = 10
num_epochs = 100
n_hidden1 = 50
n_hidden2 = 100
n_input = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = 1

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

get_mini_batch_train = GetMiniBatch(X_train, y_train, batch_size=batch_size)

def example_net(x):  
    tf.random.set_random_seed(0)  
    weights = {  
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden1])),  
        'w2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),  
        'w3': tf.Variable(tf.random_normal([n_hidden2, n_classes]))  
    }  
    biases = {  
        'b1': tf.Variable(tf.random_normal([n_hidden1])),  
        'b2': tf.Variable(tf.random_normal([n_hidden2])),  
        'b3': tf.Variable(tf.random_normal([n_classes]))  
    }  
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])  
    layer_1 = tf.nn.relu(layer_1)  
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])  
    layer_2 = tf.nn.relu(layer_2)  
    layer_output = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return layer_output

logits = example_net(X)  
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  
train_op = optimizer.minimize(loss_op)  
correct_pred = tf.equal(tf.sign(Y - 0.5), tf.sign(tf.sigmoid(logits) - 0.5))   
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
init = tf.global_variables_initializer()

with tf.Session() as sess:  
    sess.run(init)  
    for epoch in range(num_epochs):  
        total_batch = np.ceil(X_train.shape[0]/batch_size).astype(np.int64)  
        total_loss = 0  
        total_acc = 0  
        for i, (mini_batch_x, mini_batch_y) in enumerate(get_mini_batch_train):  
            sess.run(train_op, feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            total_loss += loss  
        total_loss /= n_samples  
        val_loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_val, Y: y_val})  
        print("Epoch {}, loss : {:.4f}, val_loss : {:.4f}, acc : {:.3f}".format(epoch, total_loss, val_loss, acc))  
    test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})  
    print("test_acc : {:.3f}".format(test_acc))
#Please summarize it in simple words. 

print("It is much faster than implemented CNN from scratch")
print("Main structure is same.")
print("First, weights and biases are initialized and then layers are defined.")
print("Next, loop through epochs. Training data is mini batch size.")
print("So forth, loss is calculated and weights and biases are updated. Finally network is evaluated by test dataset")

################ 3 #################
#Create a model of Iris using all three types of objective variables

df = pd.read_csv("Iris.csv")
#df = df[(df["Species"] == "Iris-versicolor") | (df["Species"] == "Iris-virginica")]
y = df["Species"]
X = df.loc[:, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
X = np.array(X)
y = np.array(y)
y[y == "Iris-versicolor"] = 0
y[y == "Iris-virginica"] = 1
y[y == "Iris-setosa"] = 2
#y = y.astype(np.int64)[:, np.newaxis]

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y = enc.fit_transform(y[:,np.newaxis])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

n_classes = 3
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

get_mini_batch_train = GetMiniBatch(X_train, y_train, batch_size=batch_size)

logits = example_net(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  
train_op = optimizer.minimize(loss_op)  
correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(tf.nn.softmax(logits) ,1))   
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
init = tf.global_variables_initializer()


with tf.Session() as sess:  
    sess.run(init)  
    for epoch in range(num_epochs):  
        total_batch = np.ceil(X_train.shape[0]/batch_size).astype(np.int64)  
        total_loss = 0  
        total_acc = 0  
        for i, (mini_batch_x, mini_batch_y) in enumerate(get_mini_batch_train):  
            sess.run(train_op, feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            total_loss += loss  
        total_loss /= n_samples  
        val_loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_val, Y: y_val})  
        print("Epoch {}, loss : {:.4f}, val_loss : {:.4f}, acc : {:.3f}".format(epoch, total_loss, val_loss, acc))  
    test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test})  
    print("test_acc : {:.3f}".format(test_acc))

print("Sigmoid function is used to binary classification problem.")
print("In other hands, multiclass problem we used softmax function.")
print("So three class classification problem, we replace sigmoid functions to softmax functions.")


################ 4 #################
#Create a model of Iris using all three types of objective variables

house_data = pd.read_csv('train.csv')
#print(house_data.head())

#data = house_data.loc[:,['GrLivArea', 'YearBuilt', 'SalePrice']]

X = house_data[['GrLivArea', 'YearBuilt']].to_numpy()
y = house_data[['SalePrice']].to_numpy()
print("Xshape:", X.shape)
print("yshape:", y.shape)
X = np.log1p(X)
y = np.log1p(y)

print("Xshape:", X.shape)
print("yshape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2)

learning_rate = 0.001
batch_size = 10
num_epochs = 50
n_hidden1 = 50
n_hidden2 = 100
n_input = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = 1

def reg_net(x):  
    tf.random.set_random_seed(0)  
    weights = {  
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden1])),  
        'w2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),  
        'w3': tf.Variable(tf.random_normal([n_hidden2, n_classes]))  
    }  
    biases = {  
        'b1': tf.Variable(tf.random_normal([n_hidden1])),  
        'b2': tf.Variable(tf.random_normal([n_hidden2])),  
        'b3': tf.Variable(tf.random_normal([n_classes]))  
    }  
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])  
    layer_1 = tf.nn.relu(layer_1)  
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])  
    layer_2 = tf.nn.relu(layer_2)  
    layer_output = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return layer_output
    
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

get_mini_batch_train = GetMiniBatch(X_train, y_train, batch_size=batch_size)

logits = reg_net(X)
loss_op = tf.reduce_mean(tf.square(logits -Y))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  
train_op = optimizer.minimize(loss_op)  
mean_squared_erro = tf.reduce_mean(tf.square(logits -Y)) 
init = tf.global_variables_initializer()


with tf.Session() as sess:  
    sess.run(init)  
    for epoch in range(num_epochs):  
        total_batch = np.ceil(X_train.shape[0]/batch_size).astype(np.int64)  
        total_loss = 0  
        total_mse = 0  
        for i, (mini_batch_x, mini_batch_y) in enumerate(get_mini_batch_train):  
            sess.run(train_op, feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            loss, mse = sess.run([loss_op, mean_squared_erro], feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            total_loss += loss  
            total_mse += mse
        total_loss /= n_samples
        total_mse /= n_samples  
        val_loss, val_mse = sess.run([loss_op, mean_squared_erro], feed_dict={X: X_val, Y: y_val})  
        print("Epoch {}, loss : {:.4f}, val_loss : {:.4f}, mse : {:.3f}".format(epoch, total_loss, val_loss, mse))  
    test_mse = sess.run(mean_squared_erro, feed_dict={X: X_test, Y: y_test})  
    print("test_mse : {:.3f}".format(test_mse))

print("Classification is the task of predicting a discrete class label. \n \
     Regression is the task of predicting a continuous quantity.")
print("Calculation of loss is different. In classification we used Cross-entropy loss besides in regression we used mean square error")
print("Also, activation is not used in regression.")


################ 5 #################
# Create a model of MNIST

(X_train,y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,784)
X_test = X_test.reshape(-1,784)

X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
X_train /= 255
X_test /= 255

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train_one_hot = enc.fit_transform(y_train[:,np.newaxis])
y_test_one_hot = enc.fit_transform(y_test[:,np.newaxis])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=0)

learning_rate = 0.001
batch_size = 10
num_epochs = 30
n_hidden1 = 50
n_hidden2 = 100
n_input = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = 10

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


get_mini_batch_train = GetMiniBatch(X_train, y_train, batch_size=batch_size)

def example_net(x):  
    tf.random.set_random_seed(0)  
    weights = {  
        'w1': tf.Variable(tf.random_normal([n_input, n_hidden1])),  
        'w2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),  
        'w3': tf.Variable(tf.random_normal([n_hidden2, n_classes]))  
    }  
    biases = {  
        'b1': tf.Variable(tf.random_normal([n_hidden1])),  
        'b2': tf.Variable(tf.random_normal([n_hidden2])),  
        'b3': tf.Variable(tf.random_normal([n_classes]))  
    }  
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])  
    layer_1 = tf.nn.relu(layer_1)  
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])  
    layer_2 = tf.nn.relu(layer_2)  
    layer_output = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return layer_output

logits = example_net(X)  
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))  
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  
train_op = optimizer.minimize(loss_op)  
correct_pred = tf.equal(tf.argmax(Y,1), tf.argmax(tf.nn.softmax(logits),1))   
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
init = tf.global_variables_initializer()

with tf.Session() as sess:  
    sess.run(init)  
    for epoch in range(num_epochs):  
        total_batch = np.ceil(X_train.shape[0]/batch_size).astype(np.int64)  
        total_loss = 0  
        total_acc = 0  
        for i, (mini_batch_x, mini_batch_y) in enumerate(get_mini_batch_train):  
            sess.run(train_op, feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: mini_batch_x, Y: mini_batch_y})  
            total_loss += loss  
        total_loss /= n_samples  
        val_loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_val, Y: y_val})  
        print("Epoch {}, loss : {:.4f}, val_loss : {:.4f}, acc : {:.3f}".format(epoch, total_loss, val_loss, acc))  
    test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test_one_hot})  
    print("test_acc : {:.3f}".format(test_acc))
#Please summarize it in simple words. 

print("In this problem, input data is an image and number of classes are 10.")
print("Hence input is an image. There are many networks which is classify images such as lenet, alexnet and so on.")
