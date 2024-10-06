import numpy as np

# Function to convert from RGB to LAB
def rgb_to_lab(src):
   
    r, g, b = src[:, :, 2], src[:, :, 1], src[:, :, 0]

    # Normalize RGB
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
   
    # Apply gamma correction
    def gamma_correction(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    height, width, channels = src.shape

    for i in range(height):
        for j in range(width):
            r[i][j]=gamma_correction(r[i][j])
            g[i][j]=gamma_correction(g[i][j])
            b[i][j]=gamma_correction(b[i][j])

   
    # Convert RGB to XYZ
    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize for the D65 illuminant
    X /= 0.95047
    Z /= 1.08883

    # Helper function for f(t)
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16 / 116)

    X=np.asarray(X)
    print(X)
    # Convert XYZ to LAB
    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))

    src[:, :, 2], src[:, :, 1], src[:, :, 0]=L,a,b
   

    return src


    