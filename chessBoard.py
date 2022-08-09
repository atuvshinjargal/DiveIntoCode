n_squares = 4
small_board_list = [1]
for _ in range(n_squares - 1):
    small_board_list.append(2*small_board_list[-1])
print("4マスの板に小麦を並べる（リスト）：{}".format(small_board_list))

import numpy as np

small_board_ndarray = np.array(small_board_list)
print("4マスの板に小麦を並べる（ndarray）：{}".format(small_board_ndarray))

small_board_ndarray2x2 = small_board_ndarray.reshape((2, 2))
print('ndarray: {}'.format(small_board_ndarray2x2))

def wheat_count_nxm(n,m):
    n_squares = n*m
    small_board_list = [1]
    for _ in range(n_squares - 1):
        small_board_list.append(2*small_board_list[-1])
    small_board_ndarray = np.array(small_board_list)
    small_board_ndarraynxm = small_board_ndarray.reshape((n, m))
    return small_board_ndarraynxm

wheatCount8x8 = wheat_count_nxm(8,8)
wheatCount8x8_copy =  wheatCount8x8
print('ndarray8x8: {}'.format(wheatCount8x8))

wheatCountSum = wheatCount8x8.sum()
print('Total number of wheat: {}'.format(wheatCountSum))

wheatCountColumnAVG = wheatCount8x8.mean(axis=0)
print(wheatCountColumnAVG)
import matplotlib.pyplot as plt

plt.xlabel("column")
plt.ylabel("number")
plt.title("number in each column")
plt.bar(np.arange(1,9),wheatCountColumnAVG)
plt.show()
print("haha:",wheatCount8x8_copy)
plt.xlabel("column")
plt.ylabel("row")
plt.title("heatmap")
plt.pcolor(wheatCount8x8_copy)
plt.show()