from ast import For


import time
import matplotlib.pyplot as plt

start = time.time()

THICKNESS = 0.00008
folded_thickness = THICKNESS
a = []  
folded_thickness = THICKNESS * (2 ** 43)
a.append(folded_thickness)

print(a)
plt.title("thickness of folded paper")
plt.xlabel("number of folds")
plt.ylabel("thickness[m]")
plt.tick_params(labelsize=20) # 軸の値に関する設定を行う
plt.plot(a,color='blue',linestyle='dashed')
plt.show()
print("厚さ： {}メートル".format(folded_thickness))

print("厚さ： {:.2f}キロメートル".format(folded_thickness / 1000))

elapsed_time = time.time() - start
print("time : {}[s]".format(elapsed_time))

start = time.time()

THICKNESS = 0.00008
folded_thickness = THICKNESS
a = []  

for x in range(43):
    a.append(folded_thickness)
    folded_thickness = folded_thickness * 2

print(a)
plt.title("thickness of folded paper")
plt.xlabel("number of folds")
plt.ylabel("thickness[m]")
plt.tick_params(labelsize=20) # 軸の値に関する設定を行う
plt.plot(a,color='blue',linestyle='dashed')
plt.show()
print("厚さ： {}メートル".format(folded_thickness))

print("厚さ： {:.2f}キロメートル".format(folded_thickness / 1000))

elapsed_time = time.time() - start
print("time : {}[s]".format(elapsed_time))
