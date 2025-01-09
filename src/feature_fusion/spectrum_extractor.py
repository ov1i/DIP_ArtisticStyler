import cv2
import numpy as np

def to_freq_dom(img):
    # Apply FFT to the image
    img_fft = np.fft.fft2(img)

    # Shift the zero frequency component to the center
    img_fft_shifted = np.fft.fftshift(img_fft)

    return img_fft_shifted

def to_spatial_dom(img_fft_shifted):
    img_fft = np.fft.ifftshift(img_fft_shifted)
    img = np.fft.ifft2(img_fft)
    img = np.abs(img)
    
    # Normalize to range 0-255
    cliped_img = np.clip(img, 0, 255).astype(np.uint8)

    return cliped_img

def get_mag_ph(img):
    # Extract magnitude and phase for a image
    mag, ph = np.abs(img), np.angle(img)

    return mag, ph

def blend_mag(enh_mag, painting_mag, alpha):
    blended_mag= (1 - alpha) * enh_mag + alpha * painting_mag
    
    return blended_mag

def reconstruct_fft(mag, ph):
    # Reconstruct the new frequency domain representation
    reconstructed_fft = mag * np.exp(1j * ph)

    return reconstructed_fft

def feature_fusion_wrapper(target_image, painting_image, alpha = 0.6):
    target_LCH = target_image[:,:,0]
    painting_LCH = painting_image[:,:,0]

    target_freq = to_freq_dom(target_LCH)
    painting_freq = to_freq_dom(painting_LCH)

    target_mag, target_ph = get_mag_ph(target_freq)
    painting_mag, painting_ph = get_mag_ph(painting_freq)
    
    mag = blend_mag(target_mag, painting_mag, alpha)
    
    freq_fusedLCH = reconstruct_fft(mag, target_ph)

    fusedLCH = to_spatial_dom(freq_fusedLCH)

    fusedImage = target_image.copy()
    fusedImage[:,:,0] = fusedLCH

    return fusedImage
