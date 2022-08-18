import matplotlib.pyplot as plt
import numpy as np

mean = [-3, 0]
cov = [[1.0, 0.8], [0.8, 1.0]] 
x = np.random.multivariate_normal(mean, cov, (500,2))
plt.plot(x[:,0], x[:,1], 'x')
plt.axis('equal')
plt.show()

plt.scatter(x[:,0,0], x[:,1,0])
plt.axis('equal')
plt.show()
plt.scatter(x[:,0,1], x[:,1,1])
plt.axis('equal')
plt.show()

plt.xlim((-5, 3))
plt.hist(x[:,0,0])
plt.show() 
plt.xlim((-5, 3))
plt.hist(x[:,0,1])
plt.show() 

fig, ax = plt.subplots()
ax.scatter(x[:,0,0],x[:,1,0], color='blue', label = '0')
ax.scatter(x[:,0,1],x[:,1,1], color='orange', label ='1')
plt.title("Dummy Data")
plt.legend()
plt.show()

combined_data_con = np.concatenate((x[:,:,0], x[:,:,1]))
print("Combined data by np.concatenate shape: {}".format(combined_data_con.shape))


combined_data_con = np.vstack((x[:,:,0], x[:,:,1]))
print("Combined data by np.vstack shape: {}".format(combined_data_con.shape))

label = np.empty((1000,1))
label[0:500] = 0
label[500:1000] = 1


combined_data_con = np.append(combined_data_con, label, axis = 1)
print(combined_data_con.shape)
print(combined_data_con)