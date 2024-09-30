import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import io, color

# Function to create a low-pass filter
def create_low_pass_filter(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    lp_filter = np.zeros((rows, cols), dtype=np.float32)
    cv, cu = np.ogrid[:rows, :cols]
    mask = (cv - crow) ** 2 + (cu - ccol) ** 2 <= radius ** 2
    lp_filter[mask] = 1
    return lp_filter

# Function to apply a frequency domain filter to an image
def apply_filter(image, filter_mask):
    # FFT of the image
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    
    # Apply the filter
    filtered_transform = f_transform_shifted * filter_mask
    
    # Inverse FFT to get the filtered image back
    filtered_image = np.abs(ifft2(ifftshift(filtered_transform)))
    return filtered_image

# Load the source image and target image
source_image = io.imread('res/test.png')
target_image = io.imread('res/monet.png')

# Create a low-pass filter based on the source image
low_pass_filter = create_low_pass_filter(source_image.shape[0:2], radius=30)

# Initialize an empty array for the filtered target image
filtered_target_image = np.zeros_like(target_image)

# Apply the filter to each color channel
for i in range(3):  # Loop through RGB channels
    # Get the channel from source and target images
    source_channel = source_image[:, :, i]
    target_channel = target_image[:, :, i]
    
    # Apply the low-pass filter to the target channel
    filtered_channel = apply_filter(target_channel, low_pass_filter)
    
    # Store the filtered channel in the output image
    filtered_target_image[:, :, i] = filtered_channel

# Display the original images and filtered image
plt.figure(figsize=(12, 6))

# Source Image
plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source_image)
plt.axis('off')

# Target Image
plt.subplot(1, 3, 2)
plt.title('Target Image')
plt.imshow(target_image)
plt.axis('off')

# Filtered Target Image
plt.subplot(1, 3, 3)
plt.title('Filtered Target Image')
plt.imshow(filtered_target_image.astype(np.uint8))
plt.axis('off')

plt.tight_layout()
plt.show()
