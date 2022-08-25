import numpy as np 
import pandas as pd
from keras.datasets import mnist
import tensorflow as tf 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from keras import metrics

################# 1 ####################
# Sharing and executing the official tutorial model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print("prediction:", predictions)
pred = tf.nn.softmax(predictions).numpy()
print(pred)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

################## 2 #####################
# (Advance assignment) Execute various methods
# Code reading

################## 3 #####################
# Learning Iris (binary classification) with Keras
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

n_features = X.shape[1]
n_classes = y.shape[1]

print("n features:", n_features)
print("n classes:", n_classes)


model = Sequential()
model.add(Dense(20, input_dim = n_features, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(n_classes, activation='sigmoid'))

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=True)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
fig.canvas.set_window_title('Learning Iris (binary classification) with Keras')
ax1.plot(history.history['loss'], label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'val loss')    
ax2.plot(history.history['accuracy'], label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'val accuracy')    
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('epochs')
ax1.set_title('Loss curve')
ax2.set_title('Accuracy ')
ax1.legend()
ax2.legend()
plt.show()
################## 4 #####################
# Learn Iris (multi-level classification) with Keras

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

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

n_features = X.shape[1]
n_classes = y.shape[1]

print("n features:", n_features)
print("n classes:", n_classes)


model = Sequential()
model.add(Dense(100, input_dim = n_features, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=True)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
fig.canvas.set_window_title('Learn Iris (multi-level classification) with Keras')
ax1.plot(history.history['loss'], label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'val loss')    
ax2.plot(history.history['accuracy'], label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'val accuracy')    
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('epochs')
ax1.set_title('Loss curve')
ax2.set_title('Accuracy ')
ax1.legend()
ax2.legend()
plt.show()

################## 5 #####################
# Learning House Prices with Keras
house_data = pd.read_csv('train.csv')
#print(house_data.head())

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

model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
# Compile model
model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =[metrics.mse])

print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=15, batch_size=32)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test mse:', score[1])



fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
fig.canvas.set_window_title('Learning House Prices with Keras')
ax1.plot(history.history['loss'], label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'val loss')    
ax2.plot(history.history['mean_squared_error'], label = 'train mse')
ax2.plot(history.history['val_mean_squared_error'], label = 'val mse')    
ax1.set_ylabel('Loss')
ax2.set_ylabel('MSE')
ax2.set_xlabel('epochs')
ax1.set_title('Loss curve')
ax2.set_title('MSE curve')
ax1.legend()
ax2.legend()
plt.show()

################## 6 #####################
# Learning MNIST with Keras

(X_train,y_train), (X_test, y_test) = mnist.load_data()

#X_train = X_train.reshape(-1,784)
#X_test = X_test.reshape(-1,784)

X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)
X_train /= 255
X_test /= 255

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
y_train_one_hot = enc.fit_transform(y_train[:,np.newaxis])
y_test_one_hot = enc.fit_transform(y_test[:,np.newaxis])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_one_hot, test_size=0.2, random_state=0)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        verbose=True)

score = model.evaluate(X_test, y_test_one_hot, verbose=0)
print('Test loss:', score[0])
print('Test acc:', score[1])

fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))
fig.canvas.set_window_title('Learning MNIST with Keras')
ax1.plot(history.history['loss'], label = 'train loss')
ax1.plot(history.history['val_loss'], label = 'val loss')    
ax2.plot(history.history['accuracy'], label = 'train accuracy')
ax2.plot(history.history['val_accuracy'], label = 'val accuracy')    
ax1.set_ylabel('Loss')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('epochs')
ax1.set_title('Loss curve')
ax2.set_title('Accuracy ')
ax1.legend()
ax2.legend()
plt.show()