import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.fft import dct

def histogram(img):
    """
    For each color channel, we loop over the image and increment the corresponding bin in the histogram
    
    :param img: the image to be histogrammed
    """
    M,N,K = img.shape
    hist = np.zeros((256,3)).astype(np.uint8)
    for k in range(K):
        for i in range(M):
            for j in range(N):
                hist[img[i,j,k],k] +=1
        
    return hist


def DCT2D(img):
    """
    It takes the 2D DCT of each color channel of the image
    
    :param img: the image to be transformed
    :return: The transformed image
    """
    transformed = np.ones(img.shape)
    for k in range(3):
        transformed[:,:,k]= dct(dct(img[:,:,k].T).T)
    return transformed


def thresh(matrix, r):
    """
    For each of the 3 channels, we find the nLargest largest coefficients, and set all the coefficients
    below that to zero
    
    :param matrix: the matrix to be thresholded
    :param r: the ratio of the number of non-zero elements to the total number of elements in the matrix
    """
    
    row,col,K = matrix.shape

    for k in range(K):
        nLargest = round(row * col * r)
        temp = matrix[:,:,k]
        coef = np.abs(np.sort(temp.flatten())[nLargest:][0])
        temp[temp<coef] = 0
        
    return matrix

def MSE_histograms(img1, img2):
    """
    It takes two images, subtracts them, squares the result, averages the result, and returns the
    average
    
    :param img1: The first image
    :param img2: the image that we want to compare to
    :return: The mean squared error between the two images.
    """
    squared = np.power(img1 - img2, 2)
    error = np.mean(squared, axis=0) 
    return  np.mean(error)

def MSE_DCT(img1, img2):
    """
    It takes the difference between the two images, squares it, and then takes the mean of the squared
    values
    
    :param img1: The first image
    :param img2: the image that we want to compare to img1
    :return: The mean squared error between the two images.
    """
    squared = np.power(img1-img2,2)
    return np.mean(squared)

def findImageHistogram(hist,path):
    """
    It takes the histogram of the image and the path to the database. It compares
    the histogram of the image to the calculated and returns the MSE error betweeen
    the two

    :param hist: The histogram of the reference image
    :param path: The path to the database of images
    :return: A dictionary with all the errors of the compared histograms
    """
    errors = {}
    for root,dirs,files in os.walk(path):
        for img in files:
            name = f'{path}/{img}'
            imageArray = np.asarray(Image.open(name))

            temp = histogram(imageArray)
            errors[img] = MSE_histograms(hist,temp)
    return errors

def findImageDCT(img,path,r):
    """
    It takes an img and the path to the database as well as the ratio of
    coefficients to keep of each DCT.It compares the dct of the image to 
    the calculated and returns the MSE error betweeen the two.

    :param img: The histogram of the reference image
    :param path: The path to the database of images
    :param r: Ratio of dct coefficients to keep
    :return: A dictionary with all the errors of the compared DCTs
    """
    errors = {}
    dcted = DCT2D(img)
    threshImg = thresh(dcted,r)
    for root,dirs,files in os.walk(path):
        for img in files:
            name = f'{path}/{img}'
            imageArray = np.asarray(Image.open(name))

            temp = DCT2D(imageArray)
            threshHold = thresh(temp,r)
            errors[img] = MSE_DCT(threshImg,threshHold)
            
    return errors

# Initialize two dictionaries for the loaded images and the histograms
# as well as the path of the test folder
images = {}
histograms = {}
path = './Images/Ex6/test'

# For every image in tests find its histogram and store it into the histogarms dictionary
for root,dirs,files in os.walk(path):
    for img in files: 
        name = f'{path}/{img}'
        imageArray = np.asarray(Image.open(name))
        images[img] = imageArray
        histograms[img] = histogram(imageArray)

# Path to the database
databasePath = './Images/Ex6/DataBase'

counter = 1

# For every image in the test folder compare it to the database and plot 
# the iamages that have matched using the DCT of the two images
plt.figure(1)
for key,img in images.items():
    errors = findImageDCT(img,databasePath,0.99)
    matched = min(errors.items(), key=lambda x: x[1])
    IMAGE = np.asarray(Image.open(f'{databasePath}/{matched[0]}'))
    plt.subplot(1*len(images.items()),2,counter)
    plt.imshow(images[key])
    plt.subplot(1*len(images.items()),2,counter+1)
    plt.imshow(IMAGE)
    counter +=2
    
# For every image in the test folder compare it to the database and plot 
# the images that have matched using the histograms of the two images
plt.figure(2)
counter = 1
for key,img in histograms.items():
    errors = findImageHistogram(img,databasePath)
    matched = min(errors.items(), key=lambda x: x[1])
    IMAGE = np.asarray(Image.open(f'{databasePath}/{matched[0]}'))
    plt.subplot(1*len(histograms.items()),2,counter)
    plt.imshow(images[key])
    plt.subplot(1*len(histograms.items()),2,counter+1)
    plt.imshow(IMAGE)
    counter +=2
plt.show()
    
