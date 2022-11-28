

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
from scipy.signal import convolve2d


def gaussian_kernel(size, sigma=1):
    """
    It creates a 2D Gaussian kernel with a given size and standard deviation
    
    :param size: The size of the kernel
    :param sigma: Standard deviation of the Gaussian kernel, defaults to 1 (optional)
    :return: A 2D array of the Gaussian kernel.
    """
    size = int(size) // 2
    x,y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma **2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def normalization(img):
    """
    It takes an image and returns the same image with all values normalized to be between 0 and 255
    
    :param img: the image to be normalized
    :return: The image is being normalized to a scale of 0 to 255.
    """
    return img / img.max() * 255

def sobel_filters(img):
    """
    It takes an image, applies the Sobel filter to it, and returns the result
    
    :param img: The image to be convolved
    :return: The gradient magnitude of the image.
    """
    Wx = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3)).astype(np.float32)
    Wy = Wx.T
    
    Fx = convolve2d(img,Wx,boundary='symm', mode='same')
    Fy = convolve2d(img,Wy, boundary='symm', mode='same')

    G = np.hypot(Fx,Fy)
    G = normalization(G)

    return G


def findFourierDescriptors(img):
    """
    It takes an image, blurs it, finds the edges, and then finds the Fourier Descriptors of the edges
    
    :param img: the image to be processed
    :return: The fourier descriptors and the edges of the image
    """
    blurred = convolve2d(img, gaussian_kernel(3,6), boundary='symm', mode='same')
    edges = sobel_filters(blurred)
    edges[edges > 35] =255
    edges[edges < 35] =0
    indices = np.where(edges == 255)
    

    M = indices[0].shape[0]
    whitePixels = np.empty(M, dtype=np.complex128)
    whitePixels.real = indices[0].T
    whitePixels.imag = indices[1].T
    FD = np.fft.fft(whitePixels)
    
    return FD,edges

def reconstructImage(descriptors,shape):
    """
    It takes the fourier descriptors and the shape of the image, and returns the image
    
    :param descriptors: the descriptors of the image
    :param shape: The shape of the image to be reconstructed
    :return: The image is being returned.
    """

    whitePixels = np.fft.fftshift(descriptors)
    middle = descriptors.size //2
    center = whitePixels[middle-500:middle+500]
    left = whitePixels[1:middle-500:2]
    right = whitePixels[middle+500::2]
    
    whitePixels = np.concatenate((left,center,right))
    whitePixels = np.fft.fftshift(whitePixels)
    whitePixels = np.fft.ifft(whitePixels)

    x = np.round(whitePixels.real).astype(np.int32)
    y = np.round(whitePixels.imag).astype(np.int32)

    img = np.zeros(shape).astype(np.uint8)
    
    for i,j in zip(np.abs(x),np.abs(y)):
        try:
            img[i,j] = 255
        except:
            pass
    return img


def enlarge(descriptors,a):
    """
    **enlarge** takes in the descriptors of an image and returns them 
    multiplied by a
    
    :param descriptors: a list of descriptors
    :param a: the amount to enlarge the image by
    :return: The descriptors are being multiplied by the value of a.
    """
    return descriptors * a

def moveImage(descriptors,dx,dy):
    """
    It takes the inverse Fourier transform of the descriptors, adds the desired displacement to the real
    and imaginary parts, and then takes the Fourier transform again
    
    :param descriptors: the descriptors of the image
    :param dx: the amount to move the image in the x direction
    :param dy: the amount to move the image in the y direction
    :return: The descriptors are being returned.
    """
    ifds = np.fft.ifft(descriptors)
    ifds.real +=dx
    ifds.imag +=dy
    return np.fft.fft(ifds)

def rotateImage(descriptors,theta):
    """
    It takes  the fourier descrtiptors of an image 
     and rotates them by theta radians
    
    :param descriptors: the fourier descrtiptors of an image
    :return: The descriptors are being multiplied by the exponential of theta.
    """

    return np.multiply(descriptors, np.exp(1j*theta))

# Load the image and convert it into an array
image = Image.open("./Images/Ex4/leaf.jpg").convert("L")
image = np.asarray(image)
M,N = image.shape

#Find the fourier descriptors of the image
fds,edges= findFourierDescriptors(image)

#Apply the moving,roation and scaling of the shape via the
# fourier descriptors
moved = moveImage(fds,40,50)
rotated = rotateImage(fds,-np.pi/2)
scaled = enlarge(fds,1.5)


# Reconstruct the original,the moved, the rotated
# and the enlarged shape 
# reconstruced = reconstructImage(fds,(M,N))
# recMoved = reconstructImage(moved,(M+40,N+50))
# recRotated = reconstructImage(rotated,(N,M))
# recScaled = reconstructImage(scaled,(round(1.5*M),round(1.5*N)))

reconstruced = reconstructImage(fds,(2*M,2*N))
recMoved = reconstructImage(moved,(2*M+40,2*N+50))
recRotated = reconstructImage(rotated,(2*N,2*M))
recScaled = reconstructImage(scaled,(round(3*M),round(3*N)))




# Plot everything
plt.figure(1)
plt.imshow(reconstruced,cmap="gray")
plt.title("Reconstructed shape using Fourier Descriptors")
plt.figure(2)
plt.imshow(recMoved,cmap="gray")
plt.title("Moving the shape by (40,50) using Fourier Descriptors")
plt.figure(3)
plt.imshow(recRotated,cmap="gray")
plt.title("Rotating the shape by 90° using Fourier Descriptors")
plt.figure(4)
plt.imshow(recScaled,cmap="gray")
plt.title("Scaling up the shape by a factor of 1.5 using Fourier Descriptors")
plt.show()