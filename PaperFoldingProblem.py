
import math

t0 = 0.00008
height_fuji = 3776
tn = t0
nFuji = 1
while tn<height_fuji:
    tn = tn * 2**1
    nFuji +=1

print('minimum number of times to fold the paper required for the thickness:', nFuji)

def compute_folding_count(distance, thickness):
    tn = thickness
    n = 1
    while tn<distance:
        tn = tn * 2**1
        n +=1
    return n
Centauri = 4.0175 * 10**16
foldCountCentauri = compute_folding_count(Centauri,0.00008)
moonDistance = 384400 * 10**3
foldCountMoon = compute_folding_count(moonDistance,0.00008)

print('fold a piece of paper to reach the nearest non-sun star:',foldCountCentauri)

def compute_length_paper(tickness,foldCount):
    l = ((math.pi*tickness) / 6) * (2**foldCount + 4) * (2**foldCount - 1)
    return l

paperLengthMoon = compute_length_paper(0.00008,foldCountMoon)
paperLengthFuji = compute_length_paper(0.00008,nFuji)
paperLengthCentauri = compute_length_paper(0.00008,foldCountCentauri)

print('length of paper needed to reach the Moon:', paperLengthMoon)
print('length of paper needed to reach the Mt. Fuji:', paperLengthFuji)
print('length of paper needed to reach the Proxima Centauri:', paperLengthCentauri)
