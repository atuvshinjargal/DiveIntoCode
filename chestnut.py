import matplotlib.pyplot as plt
import math

r=0.03175/2 #Chestnut diameter is 3.175cm
chestnut_volume = (4/3)*math.pi*r**3
system_rad = 39.5 * 1.496*10**8 #Solar system radius is 39.5 * 1.496*10^8 meter
solar_system_volume = (4/3)*math.pi*system_rad**3

print(chestnut_volume)
print(solar_system_volume)

doubled_time = 5

total_cover_time_list = []
cover_time_list = []
times=[]



def calculate_time_to_cover(solar_system_volume, oVolume, doubled_time=5):
    '''
    packing density of spheres - pi/3*sqrt(2) is about 0.740486
    To calculate number of chestnuts to fill solar system volume, I used following equation.

        number of chestnutes = 0.740486 * system_volume / chesnuts volume

    Parameters
    -----------------------------
    system_volume : int
        system volume that is covered by object (meters^3)
    
    obj_volume : int
        object volume that cover the system (meters^3)

    Returns
    ------------------------------
    total_cover_time_list : list
        List of total number of chestnuts you get that time 
    cover_time_list : list
        List of the number of chestnuts you every 5 minutes
    ------------------------------
    '''
    i = 1
    time=0
    times = []
    cover_time_list = []
    total_cover_time_list = []
    calc_volume = oVolume
    total_volume=oVolume

    n = 0.740486 * solar_system_volume / oVolume
    print('{} chestnuts fill the solar system'.format(n))
    while i < n:
        calc_volume = 2 * calc_volume
        total_volume = total_volume + calc_volume
        i = i * 2
        time= time + doubled_time
        total_cover_time_list.append(total_volume)
        cover_time_list.append(calc_volume)
        times.append(time) 

    return cover_time_list, total_cover_time_list, times


cover_time_list, total_cover_time_list, times = calculate_time_to_cover(solar_system_volume, chestnut_volume)

print("total time to cover system: {} minutes".format(times[-1]))


fig, ax = plt.subplots()
ax.plot(times,total_cover_time_list, color='red', label = 'total volume')
ax.plot(times,cover_time_list, color='green', label ='added volume')
plt.title("Volume of chestnuts")
plt.xlabel("time (min)")
plt.ylabel("volume (m^3)")
plt.legend()
plt.show()



