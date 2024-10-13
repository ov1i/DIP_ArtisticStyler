import numpy as np
import statistics as st

def subtrScalingAdd(src, painting):
      L,a,b=src[:, :, 0], src[:, :, 1], src[:, :, 2]
      Lp,ap,bp=src[:, :, 0], src[:, :, 1], src[:, :, 2]
      #subtraction
      l1=L-np.mean(L)
      a1=a-np.mean(a)
      b1=a-np.mean(b)
      #scaling
      l_scale=(np.mean(Lp)/np.mean(L))*l1
      a_scale=(np.mean(ap)/np.mean(a))*a1
      b_scale=(np.mean(bp)/np.mean(b))*b1
      #addition
      l_new=l_scale+L
      a_new=a_scale+a
      b_new=b_scale+b

      return l_new, a_new, b_new





