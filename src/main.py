import cv2 
import numpy as np
import conversion.color_space_conversions as cc

import subtr.subtr as sub
initial_img = cv2.imread("/Users/horvathdaiana/DIP_ArtisticStyler/res/test1.jpeg")
painting_img=cv2.imread("/Users/horvathdaiana/DIP_ArtisticStyler/res/oil_test2.jpeg")
img=cv2.cvtColor(initial_img,cv2.COLOR_RGB2LAB)
painting=cv2.cvtColor(painting_img, cv2.COLOR_RGB2Lab)
black_img= np.zeros((1200, 1200, 3), dtype = np.uint8)

transf_img=cc.rgb_to_lab(img)
transf_img_painting=cc.rgb_to_lab(painting)
cv2.imshow("initial image", initial_img)
cv2.imshow("painting image", painting_img)
cv2.imshow("rgb to lab initial", img)
cv2.imshow("rgb to lab painting", painting)

sub_initial_img=sub.subtrScalingAdd(img, painting)
lab_to_rgb_initial=cv2.cvtColor(sub_initial_img, cv2.COLOR_Lab2RGB)
cv2.imshow("test color matching",lab_to_rgb_initial)
cv2.waitKey(0)


cv2.imshow("test", transf_img)

cv2.waitKey(0)






