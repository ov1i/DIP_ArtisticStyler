import numpy as np

def subtrScalingAdd(src, painting):
      L,a,b=src[:, :, 0], src[:, :, 1], src[:, :, 2]
      Lp,ap,bp=painting[:, :, 0], painting[:, :, 1], painting[:, :, 2]
      #subtraction
      l1=Lp-np.mean(Lp)
      a1=ap-np.mean(ap)
      b1=bp-np.mean(bp)
      #scaling
      l_scale=(np.std(Lp)/np.std(L))*l1
      a_scale=(np.std(ap)/np.std(a))*a1
      b_scale=(np.std(bp)/np.std(b))*b1
      #addition
      l_new=np.mean(l_scale)+L
      a_new=np.mean(a_scale)+a
      b_new=np.mean(b_scale)+b

      lab_image = np.stack([l_new, a_new, b_new], axis=-1).astype(np.float32)
      return lab_image





