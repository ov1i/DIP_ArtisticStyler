import numpy as np

## Gaussian Kernel Generator
def GK_generator(size, sigma):

    if not np.isclose(6 * sigma + 1, size):
        print("Warning kernel size should be close to ", int(6 * sigma + 1), " (6 * sigma + 1)")
        print("Using recommanded size ", int(6 * sigma + 1))
        size = int(6 * sigma + 1)


    k = [[0 for _ in range(size)] for _ in range(size)]
    c = size // 2
    norm_val = 0

    for i in range(size):
        for j in range(size):
            x,y = i - c, j - c
            k[i][j] = (1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
            norm_val += k[i][j]
    
    for i in range(size):
        for j in range(size):
            k[i][j] /= norm_val

    return k