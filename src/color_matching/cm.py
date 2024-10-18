import numpy as np
import cv2

def match_colors(src, painting):
      Ls,As,Bs = cv2.split(src)
      Lp,Ap,Bp = cv2.split(painting)

      #subtraction
      Ls -= np.mean(Ls)
      As -= np.mean(As)
      Bs -= np.mean(Bs)

      #scaling
      Ls *= np.std(Lp)/np.std(Ls)
      As *= np.std(Ap)/np.std(As)
      Bs *= np.std(Bp)/np.std(Bs)

      #addition
      Ls += np.mean(Lp)
      As += np.mean(Ap)
      Bs += np.mean(Bp)

      #clip
      Ls = np.clip(Ls, 0, 100)
      As = np.clip(As, -128, 127)
      Bs = np.clip(Bs, -128, 127)

      lab_image = cv2.merge([Ls,As,Bs])

      return lab_image





