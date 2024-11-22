# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# img = cv2.imread("res/brushes/brush1.jpg")
# f = np.fft.fft2(img)

# # temp method to see the spectrum 
# f1 = np.fft.fftshift(f)
# m = np.abs(f1)
# m = np.log1p(m)
# m = 255*m/np.max(m)
# m = m.astype(np.uint8)
# # temp method to see the spectrum

# plt.figure()
# plt.imshow(m, cmap='gray')
# plt.show()