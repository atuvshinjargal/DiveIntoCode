import matplotlib.pyplot as plt

day = 1
rice = 1
list_day = []
list_rice = []
while  day<=100:
    list_day.append(day)
    list_rice.append(rice)
    day += 1
    rice *= 2

fig, ax=plt.subplots()
ax.plot(list_day,list_rice)
plt.show()


def compute_sorori_shinzaemon(n_days=100):
    list_n_grains = []
    list_total_grains = []
    rice = 1
    num = 0
    while num < n_days:
        list_n_grains.append(num)
        list_total_grains.append(rice)
        rice = rice*2
        num += 1
    return list_n_grains, list_total_grains

list_n_grains, list_total_grains = compute_sorori_shinzaemon(n_days=10)
print(list_n_grains)
print(list_total_grains)

def compute_can_live_day(rice, people):
    neededRice = 19200
    days = rice/(people*neededRice)
    return days

days = compute_can_live_day(rice, 1000000)
print(days)