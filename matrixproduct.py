'''
import markdown

html = markdown.markdown("\left(\begin{array}{cc} 0.8944272 & 0.4472136\\ -0.4472136 & -0.8944272 \end{array}\right) \left(\begin{array}{cc} 10 & 0\\ 0 & 5 \end{array}\right)")
print(html)
'''

import numpy as np
a_ndarray = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
b_ndarray = np.array([[0, 2, 1], [0, 2, -8], [2, 9, -1]])

calculated = np.matmul(a_ndarray,b_ndarray)
print(calculated)
calculated_dot = np.dot(a_ndarray,b_ndarray)
print(calculated_dot)
calculated_at = a_ndarray @ b_ndarray
print(calculated_at)

sum =0
for k in range(a_ndarray.shape[1]):
    sum = a_ndarray[0,k]*b_ndarray[k,0]
print(sum)

def product_matrix (a,b):
    product = np.empty((a.shape[0],b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(b.shape[1]):
                product[i, j] = product[i, j] + a[i,k]*b[k,j]
    print(product)

product_matrix(a_ndarray,b_ndarray)
d_ndarray = np.array([[-1, 2, 3], [4, -5, 6]])
e_ndarray = np.array([[-9, 8, 7], [6, -5, 4]])
if (d_ndarray.shape[0] == e_ndarray.shape[1]):
    product_matrix(d_ndarray,e_ndarray)
else:
    product_matrix(d_ndarray, np.transpose(e_ndarray))