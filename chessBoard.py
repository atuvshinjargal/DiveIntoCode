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

row1, row2 = np.vsplit(wheatCount8x8_copy, [4])
print('row1:',row1.shape)
print('row2:',row2.shape)
multiplyRows = row1 *row2
print('multiplyRows:',multiplyRows)

n_squares = 4
small_board_ndarray = np.array([1])
for _ in range(n_squares - 1):
    small_board_ndarray = np.append(small_board_ndarray, 2*small_board_ndarray[-1])
print("4マスの板に小麦を並べる（ndarray）：{}".format(small_board_ndarray))

n_squares = 4
indices_of_squares = np.arange(n_squares)
small_board_ndarray = 2**indices_of_squares
print("4マスの板に小麦を並べる（ndarray）：{}".format(small_board_ndarray))

a = np.array([0,1,2])

#ブロードキャストを使わない場合
b = np.array([5,5,5])
print(a + b)  # Out: [5,6,7]

#ブロードキャストを使う場合
print(a + 5)  # Out: [5,6,7]　※5が自動的に(1,3)の行列([5,5,5])に変換されている