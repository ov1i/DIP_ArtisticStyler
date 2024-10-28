import sys

import cv2 
import numpy as np

import conversion.color_space_conversions as cc
import color_matching.cm as cm
from feature_fusion.edge_enhancement import edge_enhancement_wrapper as enh_proc

print("\n\n:::::::::SCRIPT RUNNING:::::::::\n\n")

initial_img = cv2.imread("res/source/p4.jpg")
painting_img = cv2.imread("res/source/p3.jpg")

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

enhanced_img=enh_proc(cm_img_bgr)

cv2.imshow("Test Algo v1", enhanced_img)
cv2.imshow("Original Image", initial_img)

cv2.waitKey(0)

print("\n\n:::::::EXIT EXECUTED SUCCESFULLY:::::::\n\n")


