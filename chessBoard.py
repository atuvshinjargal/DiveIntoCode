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

def calculate_wheat_chess_boardcast(n_board, m_board):
    """
    a function that returns an ndarray that describes the number of wheat on the chess board of n × m squares
    using broadcast method
    Parameters 
    -----------------------
    n_board : int 
        lenght of the board
    m_board : int 
        width of the board
    
    Returns
    -----------------------
    board_array: ndarray
        wheats on the board array
    """
    n_squares = n_board * m_board
    
    indices_of_squares = np.arange(n_squares).astype(np.uint64)
    board_ndarray = 2**indices_of_squares
    #board_array = np.array(board_list)
    board_array = board_ndarray.reshape(n_board, m_board)
    return board_array

board_array = calculate_wheat_chess_boardcast(8,8) 
n_wheats = np.sum(board_array)
print("number of wheats on the {} x {} board: {}".format(8, 8, n_wheats))


def calculate_wheat_chess_ndarray(n_board, m_board):
    """
    a function that returns an ndarray that describes the number of wheat on the chess board of n × m squares
    using np.append array
    Parameters 
    -----------------------
    n_board : int 
        lenght of the board
    m_board : int 
        width of the board
    
    Returns
    -----------------------
    board_array: ndarray
        wheats on the board array
    """
    n_squares = 8 * 8
    board_ndarray = np.array([1]).astype(np.uint64)
    print("board ndarray:", board_ndarray)
    for _ in range(n_squares - 1):
      board_ndarray = np.append(board_ndarray, 2*board_ndarray[-1])
    print(board_ndarray)
    board_array = board_ndarray.reshape(8, 8)
    
    return board_array


board_array = calculate_wheat_chess_ndarray(8,8) 
print(board_array)
n_wheats = np.sum(board_array)
print("number of wheats on the {} x {} board: {}".format(8, 8, n_wheats))