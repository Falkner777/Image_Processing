
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.fft import fft2, fftshift, ifft2


def SNR_calculation(ref, noise):
    """
    The function takes in two arrays, one of which is the reference signal and the other is the noise
    signal. The function then calculates the signal to noise ratio by taking the sum of the squared
    reference signal and dividing it by the sum of the squared difference between the reference and
    noise signals
    
    :param ref: reference signal
    :param noise: the noise signal
    :return: The SNR value in dB
    """
    a = np.sum(np.power(ref,2))
    b = np.sum(np.power(ref-noise,2))
    return 10*np.log(np.divide(a,b))

def Wiener1(img,noisePower):
    """
    The function takes an image and a noise power as input, and returns the Wiener filtered image
    First it pads the image with zeros and subtracts the mean value. Then it calculates the power
    of the padded image. It calculates the Hw of the signal and removes the padding and adds the mean
    back to the signal. Finally inverts the image back to the space domain
    :param img: The image to be filtered
    :param noisePower: The power of the noise in the image
    :return: The filtered image
    """
    g = np.zeros((2*img.shape[0], 2*img.shape[1]))
    g[:img.shape[0],:img.shape[1]] = img
    mg = np.mean(img)

    g = g - mg
    Ge = fft2(g)
    Pge = powerCalculation(Ge)
    Pf = Pge - noisePower
    Hw = np.divide(Pf, Pf + noisePower)
    F_hat = np.multiply(Hw, Ge)

    fe = ifft2(F_hat)
    fe = fe[:img.shape[0],:img.shape[1]] 
    fe = fe + mg

    return fe

def Wiener2(img):
    """
    The function takes an image and a noise power as input, and returns the Wiener filtered image
    First it pads the image with zeros and subtracts the mean value. Then it calculates the power
    of the padded image. It calculates the Hw of the signal using the mean of a square in the
    high frequencies of the padded iamge.Then removes the padding and adds the mean
    back to the signal. Finally inverts the image back to the space domain
    :param img: The image to be filtered
    :param noisePower: The power of the noise in the image
    :return: The filtered image
    """
    g = np.zeros((2*img.shape[0], 2*img.shape[1]))
    g[:img.shape[0],:img.shape[1]] = img
    mg = np.mean(img)

    g = g - mg
    Ge = fft2(g)
    
    Ge_shifted = fftshift(Ge)
    noise_square =  np.mean(np.abs(Ge_shifted[25:50, 450:550]))
    Pge = powerCalculation(Ge)
    Pf = Pge - noise_square
    Hw = np.divide(Pf, Pf + noise_square)
    F_hat = np.multiply(Hw, Ge)

    fe = ifft2(F_hat)
    fe = fe[:img.shape[0],:img.shape[1]] 
    fe = fe + mg

    return fe


def powerCalculation(arr):
    """
    It takes an array of complex numbers, and returns an array of the same size, where each element is
    the square of the magnitude of the corresponding element in the input array
    
    :param arr: The array of values to be used in the power calculation
    :return: The power of the signal
    """
    return (np.abs(arr)*np.abs(arr)) 

def MSE(img1, img2):
    """
    It takes two images and returns the mean squared error between them
    
    :param img1: The first image
    :param img2: the image that we want to compare to
    :return: The mean squared error between the two images.
    """
    squared = np.power(img1 - img2, 2) / 2
    return squared

# Load the image and convert it to grayscale as well ass normalize it to 0,1
data = Image.open("./Images/Ex5/lenna.jpg").convert("L")
image = np.asarray(data)
image = image / 255
image = image[:255,:]

# Add gaussian noise as to accomplish 10dB SNR
mean = 0
sigma = 0.3258
gaussian = np.random.normal(mean, sigma, (image.shape[0],image.shape[1]))
imageWithNoise = image + gaussian
print(SNR_calculation(image,imageWithNoise))

# Apply the 2 kinds of Wiener filters
fe1 = Wiener1(imageWithNoise,sigma**2)
fe2 = Wiener2(imageWithNoise)

# Plot everything
plt.subplot(2,4,1)
plt.imshow(image,cmap='gray')
plt.title("Original image")
plt.subplot(2,4,2)
plt.imshow(imageWithNoise,cmap='gray')
plt.title("Noisy Image with 10dB SNR")
plt.subplot(2,4,3)
plt.imshow(np.abs(fe1),cmap='gray')
plt.title("After wiener knowing noise power")
plt.subplot(2,4,4)
plt.imshow(np.abs(MSE(fe1,image)),cmap='gray')
plt.title("MSE")

plt.subplot(2,4,5)
plt.imshow(image,cmap='gray')
plt.title("Original image")
plt.subplot(2,4,6)
plt.imshow(imageWithNoise,cmap='gray')
plt.title("Noisy Image with 10dB SNR")
plt.subplot(2,4,7)
plt.imshow(np.abs(fe2),cmap='gray')
plt.title("After wiener without noise power")
plt.subplot(2,4,8)
plt.imshow(np.abs(MSE(fe2,image)),cmap='gray')
plt.title("MSE")
plt.show()
