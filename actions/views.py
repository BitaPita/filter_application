
# filters_app/views.py
import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
import cv2  # Add this import at the top
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage  #  for better filtering

def apply_threshold(image, thresh_value=127):
    img_array = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_array, thresh_value, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)

def apply_invert(image):
    from PIL import ImageOps
    return ImageOps.invert(image.convert('RGB'))

def apply_sepia(image):
    width, height = image.size
    pixels = image.load()
    for py in range(height):
        for px in range(width):
            r, g, b = image.getpixel((px, py))
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            pixels[px, py] = (min(tr, 255), min(tg, 255), min(tb, 255))
    return image

def apply_find_edges(image):
    return image.filter(ImageFilter.FIND_EDGES)

def apply_emboss(image):
    return image.filter(ImageFilter.EMBOSS)
def apply_clahe(image):
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    res = clahe.apply(img_array)
    return Image.fromarray(res)

def apply_dilation(image):
    img_array = np.array(image.convert('L'))
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(img_array, kernel, iterations=1)
    return Image.fromarray(dilated)


def add_periodic_noise(image, amplitude=30, frequency=0.1):
  
    img_array = np.array(image, dtype=np.float32)
    rows, cols = img_array.shape[:2]
    
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * (x + y))
    
   
    if img_array.ndim == 3:
        sine_wave = np.expand_dims(sine_wave, axis=-1)
        
    noisy_image = img_array + sine_wave
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)


def add_speckle_noise(image, std=0.15):
   
    img_array = np.array(image, dtype=np.float32)
    
   
    noise = np.random.normal(0, std, img_array.shape)
    
  
    noisy_image = img_array + img_array * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)


def add_poisson_noise(image):
    """Add Poisson noise (Shot noise - common in low light)"""
    img_array = np.array(image, dtype=np.float32)
    
    
    img_array_clamped = np.clip(img_array, 0, None)
    
    # تولید نویز بر اساس توزیع پواسون
    noisy_image = np.random.poisson(img_array_clamped)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)

def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    img_array = np.array(image, dtype=np.float32)
    
    # تولید نویز گوسی با ابعاد تصویر
    noise = np.random.normal(mean, sigma, img_array.shape)
    
    # اضافه کردن نویز به تصویر و محدود کردن مقادیر بین 0 تا 255
    noisy_image = img_array + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_image)


def add_salt_and_pepper_noise(image, amount=0.04):
    """Add Salt and Pepper noise to the image"""
    img_array = np.array(image)
    noisy_image = img_array.copy()
  
    random_matrix = np.random.rand(*img_array.shape[:2])
  
    noisy_image[random_matrix < (amount / 2)] = 255
    
    noisy_image[random_matrix > (1 - amount / 2)] = 0
    
    return Image.fromarray(noisy_image)


def get_magnitude_spectrum(image):
    img = np.array(image)

    # convert to grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Fourier Transform
    f = np.fft.fft2(img)

    # Shift zero freq to center
    fshift = np.fft.fftshift(f)

    # Magnitude
    magnitude = np.log(1 + np.abs(fshift))

    # Normalize to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)

    return Image.fromarray(magnitude)


def get_phase_spectrum(image):
    img = np.array(image)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Phase
    phase = np.angle(fshift)

    # Normalize for visualization
    phase = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX)
    phase = phase.astype(np.uint8)

    return Image.fromarray(phase)
#-----------------------guassian----------------------------------------------------
def add_gaussian_noise_lowpass(image, mean=0, sigma=25, d0=30):
    """Add low-pass Gaussian noise (smooth noise)"""
    img_array = np.array(image, dtype=np.float32)
    
    def process_channel(channel):
        noise = np.random.normal(mean, sigma, channel.shape)
        
        noise_fft = np.fft.fft2(noise)
        noise_fft_shift = np.fft.fftshift(noise_fft)
        
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow)**2 + (j - ccol)**2)
                mask[i, j] = np.exp(-(d**2) / (2 * (d0**2)))
        
        filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
        return channel + filtered_noise
    
    if img_array.ndim == 3:
        result = np.zeros_like(img_array)
        for i in range(img_array.shape[2]):
            result[:, :, i] = process_channel(img_array[:, :, i])
        noisy_image = result
    else:
        noisy_image = process_channel(img_array)
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_gaussian_noise_highpass(image, mean=0, sigma=25, d0=30):
    """Add high-pass Gaussian noise (sharp noise)"""
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = 1 - np.exp(-(d**2) / (2 * (d0**2)))
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)
def add_gaussian_noise_highpass(image, mean=0, sigma=25, d0=30):
    """Add high-pass Gaussian noise (sharp noise)"""
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = 1 - np.exp(-(d**2) / (2 * (d0**2)))
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_gaussian_noise_bandpass(image, mean=0, sigma=25, d0=30):
    """Add band-pass Gaussian noise"""
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    # ایجاد ماسک میان‌گذر گوسی
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    d1 = d0 // 2  # فرکانس پایین‌تر
    d2 = d0       # فرکانس بالاتر
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d1 <= d <= d2:
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)
def add_gaussian_noise_notch(image, mean=0, sigma=25, d0=30, notch_width=10):
    """
    Add Gaussian noise with notch filter (band-stop)
    حذف یک محدوده فرکانسی خاص از نویز
    """
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)  # شروع با 1 (عبور همه)
    
    # حذف فرکانس‌های بین d0-notch_width تا d0+notch_width
    d1 = d0 - notch_width // 2
    d2 = d0 + notch_width // 2
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d1 <= d <= d2:
                mask[i, j] = 0 
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)
#------------------------------butterworth-----------------------------------------------------
def add_butterworth_noise_lowpass(image, mean=0, sigma=25, d0=30, n=2):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            mask[i, j] = 1 / (1 + (d / d0)**(2 * n))
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_butterworth_noise_highpass(image, mean=0, sigma=25, d0=30, n=2):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d == 0:
                mask[i, j] = 0
            else:
                mask[i, j] = 1 / (1 + (d0 / d)**(2 * n))
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_butterworth_noise_bandpass(image, mean=0, sigma=25, d0=30, n=2):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    d1 = d0 // 2
    d2 = d0
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d1 <= d <= d2:
                mask[i, j] = 1 / (1 + (d / d0)**(2 * n))
            else:
                mask[i, j] = 0
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_butterworth_noise_notch(image, mean=0, sigma=25, d0=30, notch_width=10, n=2):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)
    d1 = d0 - notch_width // 2
    d2 = d0 + notch_width // 2
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d1 <= d <= d2:
                mask[i, j] = 0
            else:
                mask[i, j] = 1 / (1 + (d / d0)**(2 * n))
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)
#---------------------------idael--------------------------------------------------------
def add_ideal_noise_lowpass(image, mean=0, sigma=25, d0=30):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d <= d0:
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_ideal_noise_highpass(image, mean=0, sigma=25, d0=30):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d >= d0:
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_ideal_noise_bandpass(image, mean=0, sigma=25, d0=30):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    d1 = d0 // 2
    d2 = d0
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d1 <= d <= d2:
                mask[i, j] = 1
            else:
                mask[i, j] = 0
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def add_ideal_noise_notch(image, mean=0, sigma=25, d0=30, notch_width=10):
    img_array = np.array(image, dtype=np.float32)
    
    if img_array.ndim == 3:
        gray = np.mean(img_array, axis=2)
        noise = np.random.normal(mean, sigma, gray.shape)
    else:
        noise = np.random.normal(mean, sigma, img_array.shape)
    
    noise_fft = np.fft.fft2(noise)
    noise_fft_shift = np.fft.fftshift(noise_fft)
    
    rows, cols = noise.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)
    d1 = d0 - notch_width // 2
    d2 = d0 + notch_width // 2
    
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if d1 <= d <= d2:
                mask[i, j] = 0
            else:
                mask[i, j] = 1
    
    filtered_noise = np.fft.ifft2(np.fft.ifftshift(noise_fft_shift * mask)).real
    
    if img_array.ndim == 3:
        noisy_image = img_array.copy()
        for i in range(3):
            noisy_image[:, :, i] += filtered_noise
    else:
        noisy_image = img_array + filtered_noise
    
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)
#-----------------------------------------------------------------------------------
def apply_filter(image, filter_type, kernel_size=3, custom_kernel=None,noise_sigma=25,
    noise_amount=0.04,
    noise_amplitude=30,
    noise_frequency=0.1,
    noise_d0=30,
    notch_width=10,
    butterworth_order=2
   ):
   
    kernel_size = int(kernel_size)
     #NOISES
   

    # Convert PIL Image to numpy array
    img_array = np.array(image)

    if filter_type == 'edge_laplacian':
        # Improved Laplacian edge detection
        if img_array.ndim == 3:  # Color image
            # Convert to grayscale first
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Normalize to 0-255
        laplacian = np.abs(laplacian)
        laplacian = np.clip(laplacian, 0, 255)
        laplacian = laplacian.astype(np.uint8)

        return Image.fromarray(laplacian)

    elif filter_type == 'edge_sobel':
        # Sobel edge detection (better for most cases)
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Sobel in X and Y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalize and scale
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
        gradient_magnitude = (gradient_magnitude * 255 / gradient_magnitude.max()).astype(np.uint8)

        return Image.fromarray(gradient_magnitude)

    elif filter_type == 'edge_canny':
        # Canny edge detection (most robust)
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Gaussian blur first (reduces noise)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection with automatic thresholding
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)

        return Image.fromarray(edges)

    elif filter_type == 'edge_prewitt':
        # Prewitt edge detection
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

        # Convolve
        prewitt_x = ndimage.convolve(gray.astype(float), kernel_x)
        prewitt_y = ndimage.convolve(gray.astype(float), kernel_y)

        # Magnitude
        magnitude = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

        # Normalize
        magnitude = np.clip(magnitude, 0, 255)
        magnitude = (magnitude * 255 / magnitude.max()).astype(np.uint8)

        return Image.fromarray(magnitude)

    elif filter_type == 'edge_log':
        # Laplacian of Gaussian (LoG) - Marr-Hildreth
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        # Find zero crossings
        laplacian = np.abs(laplacian)
        laplacian = np.clip(laplacian, 0, 255)
        laplacian = (laplacian * 255 / laplacian.max()).astype(np.uint8)

        return Image.fromarray(laplacian)

    elif filter_type == 'edge_robinson':
        # Robinson compass edge detection
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Robinson compass masks
        masks = [
            np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]]),  # North
            np.array([[0, 1, 2],
                      [-1, 0, 1],
                      [-2, -1, 0]]),  # North-East
            np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]),  # East
            np.array([[2, 1, 0],
                      [1, 0, -1],
                      [0, -1, -2]])  # South-East
        ]

        # Apply all masks and take maximum response
        responses = []
        for mask in masks:
            response = ndimage.convolve(gray.astype(float), mask)
            responses.append(np.abs(response))

        # Combine responses
        combined = np.max(responses, axis=0)

        # Normalize
        combined = np.clip(combined, 0, 255)
        combined = (combined * 255 / combined.max()).astype(np.uint8)

        return Image.fromarray(combined)

    # Keep your original filters (gaussian, box, etc.)
    elif filter_type == 'gaussian':
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))

    elif filter_type == 'box':
        return image.filter(ImageFilter.BoxBlur(radius=kernel_size / 2))

    elif filter_type == 'median':
        return image.filter(ImageFilter.MedianFilter(size=kernel_size))

    elif filter_type == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)

    elif filter_type == 'histogram_eq':
        if image.mode != 'L':
            image = image.convert('L')
        return ImageOps.equalize(image)
    elif filter_type == 'fft_magnitude':
        return get_magnitude_spectrum(image)

    elif filter_type == 'fft_phase':
        return get_phase_spectrum(image)
    
   

    elif filter_type == 'noise_gaussian':
        return add_gaussian_noise(image, sigma=noise_sigma)
    
    elif filter_type == 'noise_gaussian_lowpass':
        return add_gaussian_noise_lowpass(image, sigma=noise_sigma, d0=noise_d0)
    elif filter_type == 'noise_gaussian_highpass':
        return add_gaussian_noise_highpass(image, sigma=noise_sigma, d0=noise_d0)

    elif filter_type == 'noise_gaussian_bandpass':
        return add_gaussian_noise_bandpass(image, sigma=noise_sigma, d0=noise_d0)
    elif filter_type == 'noise_gaussian_notch':
        return add_gaussian_noise_notch(image, sigma=noise_sigma, d0=noise_d0, notch_width=notch_width)
    elif filter_type == 'butterworth_noise_lowpass':
         return add_butterworth_noise_lowpass(image, sigma=noise_sigma, d0=noise_d0, n=butterworth_order)

    elif filter_type == 'butterworth_noise_highpass':
        return add_butterworth_noise_highpass(image, sigma=noise_sigma, d0=noise_d0, n=butterworth_order)

    elif filter_type == 'butterworth_noise_bandpass':
         return add_butterworth_noise_bandpass(image, sigma=noise_sigma, d0=noise_d0, n=butterworth_order)

    elif filter_type == 'butterworth_noise_notch':
        return add_butterworth_noise_notch(image, sigma=noise_sigma, d0=noise_d0, notch_width=notch_width, n=butterworth_order)
    elif filter_type == 'ideal_noise_lowpass':
        return add_ideal_noise_lowpass(image, sigma=noise_sigma, d0=noise_d0)

    elif filter_type == 'ideal_noise_highpass':
        return add_ideal_noise_highpass(image, sigma=noise_sigma, d0=noise_d0)

    elif filter_type == 'ideal_noise_bandpass':
        return add_ideal_noise_bandpass(image, sigma=noise_sigma, d0=noise_d0)

    elif filter_type == 'ideal_noise_notch':
        return add_ideal_noise_notch(image, sigma=noise_sigma, d0=noise_d0, notch_width=notch_width)
    elif filter_type == 'noise_sp':
        return add_salt_and_pepper_noise(image, amount=noise_amount)

    elif filter_type == 'noise_periodic':
        return add_periodic_noise(
            image,
            amplitude=noise_amplitude,
            frequency=noise_frequency
        )

    

    elif filter_type == 'morph_dilation':
        # Dilation (making objects thicker)
        img_array = np.array(image)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(img_array, kernel, iterations=1)
        return Image.fromarray(dilated)

    elif filter_type == 'morph_erosion':
        # Erosion (making objects thinner)
        img_array = np.array(image)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(img_array, kernel, iterations=1)
        return Image.fromarray(eroded)

    elif filter_type == 'emboss':
        # Creative Emboss Filter
        return image.filter(ImageFilter.EMBOSS)

    elif filter_type == 'sepia':
        # Vintage Sepia effect
        img_array = np.array(image)
        if img_array.ndim == 3:
            sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            sepia_img = cv2.transform(img_array, sepia_matrix)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            return Image.fromarray(sepia_img)
        else:
            rgb_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]])
            sepia_img = cv2.transform(rgb_img, sepia_matrix)
            sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
            return Image.fromarray(sepia_img)

    elif filter_type == 'invert':
        # Color Inversion (Negative)
        if image.mode in ('RGB', 'L'):
            return ImageOps.invert(image)
        else:
            return ImageOps.invert(image.convert('RGB'))

    elif filter_type == 'threshold':
        # Binarization (Otsu's Thresholding)
        img_array = np.array(image)
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)
    
   


    elif filter_type == 'custom' and custom_kernel:
        try:
            kernel_values = list(map(float, custom_kernel.split(',')))
            n = len(kernel_values)
            size = int(np.sqrt(n))
            if size ** 2 == n:
                # Normalize kernel to avoid overflow
                kernel_sum = sum(kernel_values)
                if kernel_sum != 0:
                    kernel_values = [k / kernel_sum for k in kernel_values]
                return image.filter(ImageFilter.Kernel((size, size), kernel_values))
        except:
            pass

    return image



def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            fs = FileSystemStorage()

            # --------------------------------------------------
            # DETERMINE INPUT IMAGE (FIXED SESSION LOGIC)
            # --------------------------------------------------

            new_image_uploaded = request.FILES.get('image') is not None

            continuing = (
                    form.cleaned_data.get('continue_filtering') is True
                    and not new_image_uploaded
                    and 'history' in request.session
                    and len(request.session['history']) > 0
            )

            if continuing:
                # Use last image from history
                last_image_url = request.session['history'][-1]
                last_image_path = os.path.join(
                    settings.MEDIA_ROOT,
                    last_image_url.replace('/media/', '')
                )
                image = Image.open(last_image_path)


            else:

                # New upload → FULL RESET

                if not new_image_uploaded:
                    return render(request, 'filters_app/upload.html', {

                        'form': form,

                        'error': 'Please upload an image'

                    })



                request.session.flush()

                image_file = request.FILES['image']

                fs = FileSystemStorage()

                original_filename = fs.save(

                    f'original_{image_file.name}',

                    image_file

                )

                original_path = fs.path(original_filename)

                image = Image.open(original_path)

                request.session['history'] = [fs.url(original_filename)]

            # --------------------------------------------------
            # APPLY FILTER
            # --------------------------------------------------

            filter_type = form.cleaned_data['filter_choice']
            kernel_size = form.cleaned_data['kernel_size']
            custom_kernel = form.cleaned_data.get('custom_kernel', '')

            processed_image = apply_filter(
                image,
                filter_type,
                kernel_size,
                custom_kernel,
                noise_sigma=form.cleaned_data.get('noise_sigma') if form.cleaned_data.get('noise_sigma') is not None else 25,
                noise_amount=form.cleaned_data.get('noise_amount') if form.cleaned_data.get('noise_amount') is not None else 0.04,
                noise_amplitude=form.cleaned_data.get('noise_amplitude') if form.cleaned_data.get('noise_amplitude') is not None else 30,
                noise_frequency=form.cleaned_data.get('noise_frequency') if form.cleaned_data.get('noise_frequency') is not None else 0.1,
            )

            # --------------------------------------------------
            # SAVE PROCESSED IMAGE
            # --------------------------------------------------

            step_index = len(request.session['history'])
            processed_filename = (
                f'processed_{request.session.session_key}_{step_index}.png'
            )
            processed_path = os.path.join(
                settings.MEDIA_ROOT,
                processed_filename
            )
            processed_image.save(processed_path)

            processed_url = f'/media/{processed_filename}'

            # Append to history
            request.session['history'].append(processed_url)
            request.session.modified = True

            # --------------------------------------------------
            # CONTEXT FOR RESULTS PAGE
            # --------------------------------------------------

            history = request.session['history']

            context = {
                'original_image': history[-2] if len(history) > 1 else history[0],
                'processed_image': history[-1],
                'history': history,
                'filter_name': dict(
                    form.fields['filter_choice'].choices
                )[filter_type],
                'kernel_size': f"{kernel_size}x{kernel_size}",
                'form': ImageUploadForm(
                    initial={'continue_filtering': True}
                ),
            }

            return render(request, 'filters_app/result2.html', context)

    else:
        form = ImageUploadForm()
        request.session.pop('history', None)
        request.session.modified = True

    return render(request, 'filters_app/upload.html', {'form': form})


def redo_filtering(request):
    history = request.session.get('history', [])

    # Need at least original + one processed image
    if len(history) <= 1:
        return redirect('upload_image')

    # Remove last image
    last_image_url = history.pop()

    # Delete file from disk (optional)
    try:
        last_image_path = os.path.join(
            settings.MEDIA_ROOT,
            last_image_url.replace('/media/', '')
        )
        if os.path.exists(last_image_path):
            os.remove(last_image_path)
    except Exception:
        pass

    request.session['history'] = history
    request.session.modified = True

    # Build context exactly like upload_image does
    context = {
        'original_image': history[-2] if len(history) > 1 else history[0],
        'processed_image': history[-1],
        'history': history,
        'filter_name': 'Previous result',
        'kernel_size': '',
        'form': ImageUploadForm(initial={'continue_filtering': True}),
    }

    return render(request, 'filters_app/result2.html', context)
