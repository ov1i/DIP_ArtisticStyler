import cv2 
import numpy as np
import conversion.RgbToLab as cc
img = cv2.imread("res/test.jpeg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
transf_img=cc.rgb_to_lab(img)
mean_value=mn.mean_dev(img)
print(mean_value)
cv2.imshow("test", transf_img)
cv2.waitKey(0)



