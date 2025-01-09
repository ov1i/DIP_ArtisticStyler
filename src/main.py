import sys

import cv2 
import numpy as np
import matplotlib.pyplot as plt

import conversion.color_space_conversions as cc
import color_matching.cm as cm
from feature_fusion.edge_enhancement import edge_enhancement_wrapper as enh_proc
from feature_fusion.generators import genBrushPatterns
from feature_fusion.spectrum_extractor import feature_fusion_wrapper

print("\n\n:::::::::SCRIPT RUNNING:::::::::\n\n")

initial_img = cv2.imread("res/source/targets/t1.jpg")
initial_img = cv2.resize(initial_img, (512, 512))

painting_img = cv2.imread("res/source/paintings/p3.jpg")
painting_img = cv2.resize(painting_img, (512, 512))

if(initial_img is None):
    print("Invalid path for the source image")
    sys.exit(-1)

if(painting_img is None):
    print("Invalid path for the oil painting image")
    sys.exit(-1)

initial_img_lab = cc.bgr_rgb_to_lab(initial_img)
painting_img_lab = cc.bgr_rgb_to_lab(painting_img)

cm_img_lab = cm.match_colors(initial_img_lab, painting_img_lab)

cm_img_bgr = cc.lab_to_bgr_rgb(cm_img_lab, 1)

enhanced_img = enh_proc(cm_img_bgr)
enhanced_img_lab = cc.bgr_rgb_to_lab(enhanced_img)

brush_pic_LAB, roi_coords = genBrushPatterns(painting_img_lab[:,:,0])
brush_pic = painting_img[roi_coords[0]:roi_coords[0] + 128, roi_coords[1]:roi_coords[1] + 128]

res_LAB = feature_fusion_wrapper(enhanced_img_lab, painting_img_lab, 1)
res_BGR = cc.lab_to_bgr_rgb(res_LAB, 1)

cv2.imshow("Original Image(Target)", initial_img)
cv2.imshow("Original Image(Source)", painting_img)
# cv2.imshow("Test Algo s1", cm_img_bgr)
# cv2.imshow("Test Algo s2", enhanced_img)
# cv2.imshow("Test Algo s3", brush_pic)
cv2.imshow("Test Algo s4", res_BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n\n:::::::EXIT EXECUTED SUCCESFULLY:::::::\n\n")


