import numpy as np;

x = np.arange(-50,50,0.1)
y = 1/2*x+1

print(x)
print(y)

ndarray = np.stack((x,y)).T
print(ndarray.shape)

x_grad = x[1:] - x[:-1]
y_grad = y[1:] - y[:-1]
print(x_grad[0])
print(y_grad[0])
slope = y_grad/x_grad
print(slope[0])

import matplotlib.pyplot as plt

plt.plot(x[1:], slope, 'x')
plt.axis('equal')
plt.show()


def compute_gradient(function, x_range=(-50, 50.1, 0.1)):
    '''
    Calculate the gradient using the amount of change.
    Parameters
    -------------------------------
    function: function
       The function you want to find, the one that returns the ndarray of y when you put the ndarray of x.
     x_range: tuple
       Specify the range in the same way as the argument of np.arange ().
     Returns
     --------------------------------
     array_xy: ndarray, shape (n, 2)
       A combination of x and y. n depends on x_range.
     gradient: ndarray, shape (n-1,)
       Function gradient. One element is reduced to make a difference
    '''

    x_range = np.arange(*x_range)
    array_xy = np.stack((x_range, function)).T
    y_change = function[1:]-function[:-1]
    x_change = x_range[1:]-x_range[:-1]
    gradient = y_change/x_change
    return array_xy, gradient


def function1(array_x):
    array_x = np.arange(*array_x)
    array_y = array_x**2
    return array_y
def function2(array_x):
    array_x = np.arange(*array_x)
    array_y = 2*(array_x**2)+2**array_x
    return array_y
def function3(array_x):
    array_x = np.arange(*array_x)
    array_y = np.sin(np.sqrt(array_x))
    return array_y

def draw_plot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x[:,0], x[:,1], color='red', label = 'x vs y linear')
    ax.plot(x[1:,0], y, color='green', label ='x vs grad')
    plt.title("graph")
    plt.xlabel("x")
    plt.legend()
    plt.show()

array_x = (-50, 50.1, 0.1)
array_xy1, gradient1 = compute_gradient(function1(array_x))
draw_plot(array_xy1, gradient1)

array_xy2, gradient2 = compute_gradient(function2(array_x))
draw_plot(array_xy2, gradient2)

array_x3 = (-50, 50.1, 0.1)
array_xy3, gradient3 = compute_gradient(function3(array_x3), array_x3)
draw_plot(array_xy3, gradient3)

min1 = array_xy1[:,1].min()
armin1 = array_xy1[:,1].argmin()
print('min 1: {}'.format(min1), 'index 1: {}'.format(armin1))

min2 = array_xy2[:,1].min()
armin2 = array_xy2[:,1].argmin()
print('min 2: {}'.format(min2), 'index 2: {}'.format(armin2))

min3 = array_xy3[:,1].min()
armin3 = array_xy3[:,1].argmin()
print('min 3: {}'.format(min3), 'index 3: {}'.format(armin3))


x_range = np.arange(*array_x3)
x_range = np.delete(x_range, armin2)
y = np.delete(function3(array_x3),armin2)
y_change = y[1:]-y[:-1]
x_change = x_range[1:]-x_range[:-1]
gradient = y_change/x_change

fig, ax = plt.subplots()
ax.plot(x_range[1:], gradient, color='red', label = 'before')
ax.plot(array_xy3[1:,0], gradient3, color='green', label ='after')
plt.title("graph")
plt.xlabel("x")
plt.legend()
plt.show()