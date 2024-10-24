import cv2
import numpy as np

def edge_enhancement(src, k):
    mat=np.zeros_like(src)
    rows, cols, channels = src.shape
    for i in range(rows):
        for j in range(cols):
            for m in range(3):
                for n in range(3):
                    m1=m-2
                    n1=n-2
                    if (i+m1>=0 and j+n1>=0 and i+m1<rows and j+n1<cols):
                        mat[i][j]+=k[m][n]*src[i+m1][j+n1]
    return mat                      

                    
                 
img=cv2.imread("res/source/p1.jpg")
k=[[0 ,1, 0], [1, -4, 1], [0 ,1, 0]]
img=edge_enhancement(img, k)
cv2.imshow(img)
cv2.waitKey(0)

            
                


            