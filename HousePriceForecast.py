import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

house_data = pd.read_csv('train.csv')
#print(data.head())

# Selection of features for practice

data = house_data.loc[:,['GrLivArea', 'YearBuilt', 'SalePrice']]

print(data.loc[:,'GrLivArea'].isnull().sum())
print(data.loc[:,'YearBuilt'].isnull().sum())
print(data.loc[:,'SalePrice'].isnull().sum())

X = data.loc[:,['GrLivArea', 'YearBuilt']].to_numpy()
y = data.loc[:, 'SalePrice'].to_numpy()

# Estimation and evaluation by linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("root mean square error:{}".format(mse))
mse = mean_squared_error(y_test, y_pred, squared= False)
print("root mean square error:{}".format(mse))

fig, ax = plt.subplots()
ax.scatter(y_pred, y_test)
ax.plot((0, 400000), (0, 500000), linestyle='dashed', color='red')
ax.set_xlabel('predicted value')
ax.set_ylabel('actual value')
plt.show()
 
# Comparison of methods

svr = SVR(kernel = 'poly')
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, svr_pred, squared= False)
print("root mean square error:{}".format(mse))


regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)
reg_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, reg_pred, squared= False)
print("root mean square error:{}".format(mse))


rreg = RandomForestRegressor()
rreg.fit(X_train, y_train)
rreg_pred = rreg.predict(X_test)
mse = mean_squared_error(y_test, rreg_pred, squared= False)
print("root mean square error:{}".format(mse))


fig, ax = plt.subplots(2, 2)
ax[0][0].scatter(y_pred, y_test)
ax[0][0].plot((0, 400000), (0, 500000), linestyle='dashed', color='red')
ax[0][0].set_xlabel('predicted value')
ax[0][0].set_ylabel('actual value')
ax[0][0].title.set_text('Linear Regression')
ax[0][1].scatter(svr_pred, y_test)
ax[0][1].plot((0, 400000), (0, 500000), linestyle='dashed', color='red')
ax[0][1].set_xlabel('predicted value')
ax[0][1].set_ylabel('actual value')
ax[0][1].title.set_text('SVR')
ax[1][0].plot((0, 400000), (0, 500000), linestyle='dashed', color='red')
ax[1][0].scatter(reg_pred, y_test)
ax[1][0].set_xlabel('predicted value')
ax[1][0].set_ylabel('actual value')
ax[1][0].title.set_text('Decision Tree Regressor')
ax[1][1].scatter(rreg_pred, y_test)
ax[1][1].plot((0, 400000), (0, 500000), linestyle='dashed', color='red')
ax[1][1].set_xlabel('predicted value')
ax[1][1].set_ylabel('actual value')
ax[1][1].title.set_text('Random Forest Regressor')
plt.show()

print("SVM is the worest one while linear regression is the best") 

#  (Advance task) Learning using other features

data = house_data.loc[:,['GrLivArea', 'YearBuilt' , 'LotArea','SalePrice']]

print(data.loc[:,'GrLivArea'].isnull().sum())
print(data.loc[:,'YearBuilt'].isnull().sum())
print(data.loc[:,'SalePrice'].isnull().sum())
print(data.loc[:,'LotArea'].isnull().sum())

X = data.loc[:,['GrLivArea', 'YearBuilt', 'LotArea']].to_numpy()
y = data.loc[:, 'SalePrice'].to_numpy()

# Estimation and evaluation by linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Feature added: root mean square error:{}".format(mse))
mse = mean_squared_error(y_test, y_pred, squared= False)
print("Feature added: root mean square error:{}".format(mse))

fig, ax = plt.subplots()
ax.scatter(y_pred, y_test)
ax.plot((0, 400000), (0, 500000), linestyle='dashed', color='red')
ax.set_xlabel('predicted value')
ax.set_ylabel('actual value')
plt.show()