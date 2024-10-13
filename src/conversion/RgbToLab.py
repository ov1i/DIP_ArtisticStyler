import numpy as np
import cv2

# Function to convert from RGB to LAB
def rgb_to_lab(src):
   
    r, g, b = src[:, :, 0], src[:, :, 1], src[:, :, 2]

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

    lab_image = np.stack([L, a, b], axis=-1).astype(np.float32)

    return lab_image


    