import numpy as np
import cv2
import statistics as st

def mean_std_dev(src, mean):
    L,a,b=src[:, :, 0], src[:, :, 1], src[:, :, 2]
    ml, ma, mb=np.mean(L) , np.mean(a),  np.mean(b)
    return ml, ma, mb, np.std(L), np.std(a), np.std(b)



    

     