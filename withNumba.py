import numba  # We added these two lines for a 500x speedup

@numba.jit    # We added these two lines for a 500x speedup
def sum(x):
    total = 0
    for i in range(x.shape[0]):
        total += x[i]
    return total