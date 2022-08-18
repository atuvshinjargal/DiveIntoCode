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

plt.hist(x[:,0,0], x[:,1,0])
plt.axis('equal')
plt.show()