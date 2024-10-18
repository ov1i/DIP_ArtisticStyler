import numpy as np
import cv2

# Function to convert from LAB to RGB
def lab_to_bgr_rgb(src, bgr = 1):
    # Separate LAB channels
    L, a, b = cv2.split(src)

    # Normalize LAB values (L in [0, 100], a and b in [-128, 127])
    Y = (L + 16) / 116
    X = a / 500 + Y
    Z = Y - b / 200

    # Reverse normalize XYZ for the D65 illuminant
    X = X ** 3 if np.any(X > 0.206897) else (X - 16 / 116) / 7.787
    Y = Y ** 3 if np.any(Y > 0.206897) else (Y - 16 / 116) / 7.787
    Z = Z ** 3 if np.any(Z > 0.206897) else (Z - 16 / 116) / 7.787

    # Normalize for D65 illuminant
    X *= 0.95047
    Z *= 1.08883

    # Convert XYZ to linear RGB
    r = X * 3.2404542 + Y * -1.5371385 + Z * -0.4985314
    g = X * -0.9692660 + Y * 1.8760108 + Z * 0.0415560
    b = X * 0.0556434 + Y * -0.2040259 + Z * 1.0572252

    # Clip to the range [0, 1] before gamma correction
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    # Inverse gamma correction (sRGB)
    def inverse_gamma_correction(c):
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)

    r = inverse_gamma_correction(r)
    g = inverse_gamma_correction(g)
    b = inverse_gamma_correction(b)

    # Merge resulted data into a image with 3 color ch
    if(bgr == 0):
        res = cv2.merge([r,g,b])
    elif(bgr == 1): 
        res = cv2.merge([b,g,r])
    else:
        print("Warning invalid parameter passed!")
        print("Continuing with default value")
        res = cv2.merge([b,g,r])
        
    # Convert to uint8 values (unsigned 8 bit integers for each channel value range(0-255))
    res = (res * 255).astype(np.uint8)

    return res


def bgr_rgb_to_lab(src, bgr = 1):

    if(bgr == 1):
        b, g, r = cv2.split(src)
    elif(bgr == 0):
        r, g, b = cv2.split(src)
    else:
        print("Warning invalid parameter passed!")
        print("Continuing with default value")
        res = cv2.merge([b,g,r])

    # # Normalize RGB
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
   
     # Apply gamma correction
    def gamma_correction(c):
        return np.where(c <= 0.04045 , c/12.92, ((c + 0.055) / 1.055) ** 2.4)
    
    r=gamma_correction(r)
    g=gamma_correction(g)
    b=gamma_correction(b)
    
   
     # Convert RGB to XYZ
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041



     # Normalize for the D65 illuminant
    X /= 0.95047
    Z /= 1.08883

    # # Normalize for the D50 illuminant
    # #X /= 0.95047
    # #Z /= 1.08883
    # #TODO: to be added(check wiki)

    # # Helper function for f(t)
    def f(t):
        return np.where(t>0.008856, t**(1/3),(7.787 * t + 16 / 116 ))

    
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))

    L = np.clip(L, 0, 100)
    a = np.clip(a, -128, 127)
    b = np.clip(b, -128, 127)

    res = cv2.merge([L,a,b])

    return res
