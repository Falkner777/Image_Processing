import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


def SNR_calculation(ref, noise):
    """
    The function takes two inputs, the reference signal and the noise signal, and returns the SNR value

    :param ref: reference signal
    :param noise: the noise signal
    :return: The SNR value in dB
    """
    a = np.sum(np.power(ref, 2))
    b = np.sum(np.power(ref-noise, 2))
    return 10*np.log(np.divide(a, b))


def movingAverage(arr, window):
    """
    For each window of size `window` in the array `arr`, calculate the mean of the values in that window

    :param arr: the array to be averaged
    :param window: the size of the window to average over
    :return: The filtered image using the moving average filter
    """
    reps = arr.shape[0] - window + 1
    avg = np.ones((reps, reps))
    for i in range(reps):
        for j in range(reps):
            avg[i][j] = np.mean(arr[j:j+window, i:i+window])
    return avg.T


def movingMedian(arr, window):
    """
    For each window of size `window` in the array `arr`, compute the median of the values in that window

    :param arr: the array to be smoothed
    :param window: the size of the window to use for the median filter
    :return: The filtered image using the median filter
    """
    reps = arr.shape[0] - window + 1
    med = np.ones((reps, reps))
    for i in range(reps):
        for j in range(reps):
            med[i][j] = np.median(arr[j:j+window, i:i+window])
    return med.T


def salt_n_pepper(img, percentage):
    """
    We're going to randomly select a certain percentage of pixels in the image, and randomly set them to
    either 0 or 1

    :param img: the image to be salted and peppered
    :param percentage: the percentage of the image that will be salt and peppered
    :return: A copy of the image with a percentage of the pixels randomly set to either 0 or 1.
    """
    salted = np.copy(img)

    total_amount = int(np.ceil(percentage * salted.size))

    for i in range(total_amount):
        decision = np.random.uniform(0, 1)
        x = np.random.randint(0, img.shape[0])
        y = np.random.randint(0, img.shape[1])

        if decision > 0.5:
            salted[x, y] = 1
        else:
            salted[x, y] = 0

    return salted


# Load the data
data = loadmat("./Images/Ex3/tiger.mat")
image = np.array(data["tiger"])

# Add white noise to the image
mean = 0
sigma = 0.2033
gaussian = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))
imageWithNoise = image + gaussian

# Set the window size and apply the two filters
windowSize = 3
filteredMA_gaussian = movingAverage(imageWithNoise, windowSize)
filteredMM_gaussian = movingMedian(imageWithNoise, windowSize)

# Add salt and pepper and apply the two filters
salted = salt_n_pepper(image, 0.2)
filteredMA_sp = movingAverage(salted, windowSize)
filteredMM_sp = movingMedian(salted, windowSize)

# Add both kinds of noise
saltNGauss = salt_n_pepper(imageWithNoise, 0.2)

# Apply the 2 filters with different order
firstFilterM = movingMedian(saltNGauss, windowSize)
secondFilterA = movingAverage(firstFilterM, windowSize)

firstFilterA = movingAverage(saltNGauss, windowSize)
secondFilterM = movingMedian(firstFilterA, windowSize)

# Plot everything
plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(imageWithNoise, cmap="gray")
plt.title("Image with 15dB white Gaussian Noise")

plt.subplot(2, 2, 3)
plt.imshow(filteredMA_gaussian, cmap="gray")
plt.title(f"Image with {windowSize}x{windowSize} Moving Average Filter")

plt.subplot(2, 2, 4)
plt.imshow(filteredMM_gaussian, cmap="gray")
plt.title(f"Image with {windowSize}x{windowSize} Moving Median Filter")

plt.figure(2)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(salted, cmap="gray")
plt.title("Image with 20% salt and pepper noise")

plt.subplot(2, 2, 3)
plt.imshow(filteredMA_sp, cmap="gray")
plt.title(f"Image with {windowSize}x{windowSize} Moving Average Filter")

plt.subplot(2, 2, 4)
plt.imshow(filteredMM_sp, cmap="gray")
plt.title(f"Image with {windowSize}x{windowSize} Moving Median Filter")

plt.figure(3)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(salted, cmap="gray")
plt.title("Image with both kinds of noise")

plt.subplot(2, 2, 3)
plt.imshow(firstFilterM, cmap="gray")
plt.title(f"First filter: Moving median")

plt.subplot(2, 2, 4)
plt.imshow(secondFilterA, cmap="gray")
plt.title(f"Second filter: Moving average")

plt.figure(4)
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(salted, cmap="gray")
plt.title("Image with both kinds of noise")

plt.subplot(2, 2, 3)
plt.imshow(firstFilterA, cmap="gray")
plt.title(f"First filter: Moving average")

plt.subplot(2, 2, 4)
plt.imshow(secondFilterM, cmap="gray")
plt.title(f"Second filter: Moving median")
plt.show()
