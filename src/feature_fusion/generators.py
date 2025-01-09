import numpy as np

## Gaussian Kernel Generator
def GK_generator(size, sigma, rec = 0):

    if not np.isclose(6 * sigma + 1, size):
        print("Warning kernel size should be close to ", int(6 * sigma + 1), " (6 * sigma + 1)")
        if rec == 1:
            print("Recommanded flag ON")
            print("Using recommanded size ", int(6 * sigma + 1))
            size = int(6 * sigma + 1)
        elif rec == 0:
            print("Recommanded flag OFF")
            print("Continuing with the requested size")



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

    return np.array(k)

def GK_separator(kernel, size):
    horiz_kernel = np.sum(kernel, axis=0)
    vert_kernel = np.sum(kernel, axis=1)
    
    return horiz_kernel, vert_kernel

def compute_sd(roi):
    return np.std(roi)

def genBrushPatterns(image, window_size=(128, 128)):
    image_height, image_width = image.shape[:2]
    window_height, window_width = window_size

    min_sd = float('inf')

    min_sd_ROI = None
    roi_coords = None

    # Slide the window across the image
    for i in range(0, image_height - window_height, window_height):
        for j in range(0, image_width - window_width, window_width):
            roi = image[i:i + window_height, j:j + window_width]

            # Compute the standard deviation of the current ROI
            sd = compute_sd(roi)

            # Update minimum SD
            if sd < min_sd:
                min_sd = sd
                min_sd_ROI = roi
                roi_coords = (i, j)

    #return the coord of the BRUSH pattern and the BRUSH pattern L ch
    return min_sd_ROI, roi_coords 