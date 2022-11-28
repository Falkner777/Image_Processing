import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def linearTransform(oldMax, oldMin, newMax, newMin, value):
    """
    > Given a value, a minimum and maximum value for the old range,
    and a minimum and maximum value for
    the new range, return the value in the new range

    :param oldMax: The maximum value of the old range
    :param oldMin: The minimum value of the old range
    :param newMax: The maximum value of the new range
    :param newMin: The minimum value of the new range
    :param value: the value to be transformed
    :return: The value of the linear transformation.
    """
    return ((value - oldMin) / (oldMax - oldMin)) \
        * (newMax - newMin) + newMin


def myFFTShift(arr, complexx=0):
    """
    It multiplies the array by $(-1)^{i+j}$ where i and j are the row and column indices of the
    array
    
    :param arr: the array to be shifted
    :param complexx: 0 for real, 1 for complex, defaults to 0 (optional)
    :return: The shifted array.
    """
    M,N = arr.shape
    if complexx:
        shifted = np.zeros(arr.shape).astype(np.complex128)
    else:
        shifted = np.zeros(arr.shape)
    for i in range(M):
        for j in range(N):
            shifted[i, j] = (-1)**(i+j) * arr[i, j]
    return shifted


def my2DFFT(arr):
    """
    We first take the 1D FFT of each row, then take the 1D FFT of each column

    :param arr: the 2D array to be transformed
    :return: The 2D Fourier transform of the input matrix.
    """
    size = arr.shape[0]
    transformed = np.zeros((size, size)).astype(np.complex128)
    for i in range(size):
        transformed[i, :] = np.fft.fft(arr[i, :])
    for i in range(size):
        transformed[:, i] = np.fft.fft(transformed[:, i])
    return transformed


def my2DIFFT(arr):
    """
    We take the inverse Fourier transform of each row, then take
    the inverse Fourier transform of each
    column

    :param arr: the array to be transformed
    :return: The 2D inverse Fourier transform of the input matrix.
    """
    size = arr.shape[0]
    transformed = np.zeros((size, size)).astype(np.complex128)
    for i in range(size):
        transformed[i, :] = np.fft.ifft(arr[i, :])
    for i in range(size):
        transformed[:, i] = np.fft.ifft(transformed[:, i])

    return transformed


def idealLowPass(data, Do):
    """
    It takes a 2D array and a cutoff frequency, and returns a 2D array with all frequencies above the
    cutoff frequency set to zero
    
    :param data: The image data
    :param Do: Cutoff frequency
    :return: The filtered image.
    """
    center = data.shape[0] // 2
    filtered = np.zeros(data.shape).astype(np.complex128)
    filtered[center-Do:center+Do, center-Do:center +
             Do] = data[center-Do:center+Do, center-Do:center+Do]
    return filtered


# Load image and convert it into numpy matrix
image = Image.open("./Images/Ex1/moon.jpg").convert("L")
data = np.asarray(image)

# Get minimum and maximum value of the matrix in order to appply linear transform
maxValue = np.max(data)
minValue = np.min(data)

# Linear transformation and unsigned int conversion
data = linearTransform(maxValue, minValue, 255, 0, data)
data = data.astype(np.uint8)

# Shift the frequencies to the center of the matrix
shifted = myFFTShift(data)
# Aply 2D FFT using 1D FFT
freq = my2DFFT(shifted)

# Plot the fft of the image
plt.figure(1)
plt.imshow(np.log(np.abs(freq)), cmap="gray")
plt.title("2D FFT of moon.jpeg")


# Application of the ideal filter
filtered = idealLowPass(freq, 50)
plt.figure(2)
plt.imshow(np.abs(filtered), cmap='gray')
plt.title("Ideal low pass filter")
plt.show()

# Invert the signal with 2D IFFT and apply the shift
invertedSignal = my2DIFFT(filtered)
shifted_freqency = np.abs(myFFTShift(invertedSignal, 1)).astype(np.uint8)


# Plot the filtered and the original image
plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(data, cmap='gray')
plt.title("Image before filtering")
plt.subplot(1, 2, 2)
plt.imshow(shifted_freqency, cmap='gray')
plt.title("Image after filtering with ideal low pass filter")
plt.show()
