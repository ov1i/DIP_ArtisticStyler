import cv2
import numpy as np
from  feature_fusion import generators as g

def convolution(src, k, edge_flag=0):
    rows, cols, channels= src.shape
    sr, sg, sb=cv2.split(src)

    target = np.zeros_like(src, dtype=np.int32)
    tr, tg, tb=cv2.split(target)

    k_height, k_width = k.shape
    kh, kw = k_height // 2, k_width // 2 

    for i in range(rows):
        for j in range(cols):
            for m in range(-kh, kh + 1):
                for n in range(-kw, kw + 1):
                    if 0 <= i + m < rows and 0 <= j + n < cols:
                        tr[i, j] += k[m + kh, n + kw] * sr[i + m, j + n]
                        tg[i, j] += k[m + kh, n + kw] * sg[i + m, j + n]
                        tb[i, j] += k[m + kh, n + kw] * sb[i + m, j + n]

    target=cv2.merge([tr, tg, tb])
    target = np.clip(target, 0, 255)

    return target.astype(np.uint8)
def edge_enhancement(src, w, prc_img):
    prc_img*=w
    src-=prc_img
    w=1-w
    return src/w
def edge_enhancement_wrapper(src):
    k=g.GK_generator(5, 1)
    blurred_img = convolution(src, k).astype(np.float32)
    enhanced_img=edge_enhancement(src.astype(np.float32), 0.6, blurred_img)
    return enhanced_img


