
import numpy as np
from scipy.fft import dct,idct
from scipy.io import loadmat
from matplotlib import pyplot as plt



def DCT2D(img, window=32):
    """
    It takes an image and breaks it into blocks of size  times 32$, then applies the 2D DCT to each
    block using the row-column method
    
    :param img: the image to be transformed
    :param window: the size of the block to be transformed, defaults to 32 (optional)
    :return: The 2D DCT of the image.
    """
    transformed = np.ones(img.shape)
    reps = img.shape[0] // window
    for i in range(reps):
        for j in range(reps):
            block = img[i*window:(i+1)*window, j*window:(j+1)*window]
            transformed[i*window:(i+1)*window,j * window:(j+1) *window] = dct(dct(block.T).T)
    return transformed

def IDCT2D(img, window=32):
    """
    It takes an image and breaks it into blocks of size  \times 32$, then applies the 2D IDCT to each
    using the row-column method
    
    :param img: the image to be transformed
    :param window: the size of the block to be transformed, defaults to 32 (optional)
    :return: The transformed image
    """
    transformed = np.ones(img.shape)
    reps = img.shape[0] // window
    for i in range(reps):
        for j in range(reps):
            block = img[i*window:(i+1)*window, j*window:(j+1)*window]
            transformed[i*window:(i+1)*window,j * window:(j+1) *window] = idct(idct(block.T).T)
    return transformed


def zigzagZonal(matrix, r):
    """
    It loops through the matrix, starting from the top left corner, and going down the diagonal, and
    then going down the next diagonal, and so on. 
    It does this by looping through the rows and columns, and then using the `min` function to determine
    the number of elements in each diagonal. 
    The `min` function is used because the number of elements in each diagonal decreases as we move down
    the matrix. 
    The `max` function is used to determine the starting column of each diagonal. 
    The `round` function is used to determine the number of elements to be zeroed out. 
    The `counter` variable is used to keep track of the number of elements that have been zeroed out. 
    The `selection` variable is used to determine the number of elements to be zeroed out. 
    The `row` and `col` variables are used to determine the number of rows and column of
    the matrix
    
    :param matrix: the matrix to be zigzagged
    :param r: the percentage of the matrix that you want to be zeroed out
    :return: The matrix with the selected number of elements set to zero.
    """
    
    row = matrix.shape[0]
    col = matrix.shape[1]
    selection = round(row * col * r)
    counter = 0

    for line in range(1, (row + col)):
       
        start_col = max(0, line - row)

        count = min(line, (col - start_col), row)
        
        for j in range(0, count):
            counter +=1
            if counter > selection:
                matrix[min(row, line) - j - 1][start_col + j] = 0
        
    return matrix

def thresh(matrix, r):
    """
    It takes a matrix and a ratio r, and returns a matrix with all elements less than the r-th largest
    element set to 0.
    
    :param matrix: the matrix to be thresholded
    :param r: the ratio of the number of non-zero elements to the total number of elements in the matrix
    :return: The matrix with the values less than the coef set to 0
    """
    
    row = matrix.shape[0]
    col = matrix.shape[1]
    nLargest = round(row * col * r)
    coef = np.abs(np.sort(matrix.flatten())[nLargest:][0])
    matrix[np.abs(matrix) < coef] =0
    return matrix

  
def zonalCompression(img,r,window=32):
    """
    It takes an image, divides it into  \times 32$ blocks, and then compresses each block using the
    zigzag zonal compression algorithm
    
    :param img: the image to be compressed
    :param r: the number of coefficients to keep
    :param window: the size of the square blocks that we will be compressing, defaults to 32 (optional)
    :return: The compressed image
    """
    compressed = np.ones(img.shape)
    reps = image.shape[0] // window

    for i in range(reps):
        for j in range(reps):
            compressed[i*window:(i+1)*window,j * window:(j+1) *window] = zigzagZonal(img[i*window:(i+1)*window,j * window:(j+1) *window],r)

    return compressed

def thresholdCompression(img,r,window=32):
    """
    It takes an image and a threshold value, and returns a compressed version of the image where all
    values below the threshold are set to 0 and all values above the threshold are left intact
    
    :param img: the image to be compressed
    :param r: the threshold value
    :param window: the size of the window to be compressed, defaults to 32 (optional)
    :return: The compressed image
    """
    compressed = np.ones(img.shape)
    reps = image.shape[0] // window

    for i in range(reps):
        for j in range(reps):
            compressed[i*window:(i+1)*window,j * window:(j+1) *window] = thresh(img[i*window:(i+1)*window,j * window:(j+1) *window],r)

    return compressed
    
def MSE(img1, img2):
    """
    It takes two images and returns the mean squared error between them
    
    :param img1: The first image
    :param img2: the image that we want to compare to
    :return: The mean squared error between the two images.
    """
    squared = np.power(img1 - img2, 2) / 2
    return squared

#Load the image and convert it into a matrix 
data = loadmat("./Images/Ex2/barbara.mat")
image = np.array(data["barbara"])

#Splitting the RGB matrices into three seperate
red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

#Initializing lists for the in between stages
#before the compression
colors = [red,green,blue]
dcted5 = []
dcted50 = []
compressed_zonal5 = []
compressed_thresh5 = []
compressed_zonal50 = []
compressed_thresh50 = []
idcted_zonal5 = []
idcted_thresh5 = []
idcted_zonal50 = []
idcted_thresh50 = []
# Apply the DCT to every 32x32 block and
# append them to the list mentioned above
for color in colors:
     dcted5.append(DCT2D(color))
     dcted50.append(DCT2D(color))
# For every DCT apply the 2 different compressions
for comp in dcted5:
    compressed_zonal5.append(zonalCompression(comp,0.05))
    compressed_thresh5.append(thresholdCompression(comp,0.05))
for comp in dcted50:
    compressed_zonal50.append(zonalCompression(comp,0.5))
    compressed_thresh50.append(thresholdCompression(comp,0.5))

# For every compressed DCT invert the image back to the original 
for i,j in zip(compressed_zonal5,compressed_thresh5):
    idcted_zonal5.append(IDCT2D(i))
    idcted_thresh5.append(IDCT2D(i))   
for i,j in zip(compressed_zonal50,compressed_thresh50):
    idcted_zonal50.append(IDCT2D(i))
    idcted_thresh50.append(IDCT2D(i))  
#Initialize empty arrays of size (256,256,3) to merge the R,G,B compressed images
compressedImage_zonal5 = np.ones((256,256,3)).astype(np.uint8)
compressedImage_thresh5 = np.ones((256,256,3)).astype(np.uint8)
compressedImage_zonal50 = np.ones((256,256,3)).astype(np.uint8)
compressedImage_thresh50 = np.ones((256,256,3)).astype(np.uint8)
#Merge the R,G,B
for i in range(3):
    compressedImage_zonal5[:,:,i] = idcted_zonal5[i].astype(np.uint8)
    compressedImage_thresh5[:,:,i] = idcted_thresh5[i].astype(np.uint8)
    compressedImage_zonal50[:,:,i] = idcted_zonal50[i].astype(np.uint8)
    compressedImage_thresh50[:,:,i] = idcted_thresh50[i].astype(np.uint8)

#Calculate the errors
zonalError5 = MSE(compressedImage_zonal5,image).astype(np.uint8)
threshError5 = MSE(compressedImage_thresh5,image).astype(np.uint8)
zonalError50 = MSE(compressedImage_zonal50,image).astype(np.uint8)
threshError50 = MSE(compressedImage_thresh50,image).astype(np.uint8)

#Plot everything
plt.figure(1)
plt.subplot(3,2,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(3,2,3)
plt.imshow(compressedImage_zonal5)
plt.title("Zonal compression 5%")

plt.subplot(3,2,5)
plt.imshow(zonalError5)
plt.title("Zonal MSE 5%")

plt.subplot(3,2,2)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(3,2,4)
plt.imshow(compressedImage_zonal50)
plt.title("Zonal compression 50%")

plt.subplot(3,2,6)
plt.imshow(zonalError50)
plt.title("Zonal MSE 50%")

plt.figure(2)
#Plot everything
plt.subplot(3,2,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(3,2,3)
plt.imshow(compressedImage_thresh5)
plt.title("Threshold compression 5%")

plt.subplot(3,2,5)
plt.imshow(threshError5)
plt.title("Threshold MSE 5%")

plt.subplot(3,2,2)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(3,2,4)
plt.imshow(compressedImage_thresh50)
plt.title("Threshold compression 50%")

plt.subplot(3,2,6)
plt.imshow(threshError50)
plt.title("Threshold MSE 50%")

plt.show()