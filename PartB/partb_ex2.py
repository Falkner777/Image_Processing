
import numpy as np
import struct as st
import matplotlib.pyplot as plt
import keras
from skimage.feature import hog


from PIL import Image
from sklearn import metrics,svm


def readImages(filename):
    with open(filename,'rb') as f:
        magic, size = st.unpack(">II", f.read(8))
        nrows, ncols = st.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
    return data

def readLabels(filename):
    with open(filename,'rb') as f:
        magic, size = st.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size,))
    return data 


def resizeData(data):
    newSize = (64,128)
    n = data.shape[0]
    newData = data.reshape((n,28,28))
    resizedImages = np.zeros((n,newSize[1],newSize[0]))
    for i in range(n):
        img = Image.fromarray(newData[i, :, :]).resize(newSize)
        resizedImages[i, :, :] = np.asarray(img)
    
    return resizedImages

def getHogDescriptors(data,pixels,cells):
    n = 128 // pixels
    m = 64 // pixels
    hogDescriptors = np.zeros((data.shape[0],n*m*cells*9))

    for i in range(data.shape[0]):
        hogDescriptors[i,:] = hog(data[i,:], orientations=9, pixels_per_cell=(pixels, pixels),
                	cells_per_block=(cells, cells),block_norm="L1")
    return hogDescriptors


def makeConfusionMatrix(labels,predictions):
    confusionMatrix = np.zeros((10,10),dtype=np.int32)
    
    for i,j in zip(labels,predictions):
        confusionMatrix[i][j]+=1
    
    return confusionMatrix


trainImages = "./MnistDataset/train-images.idx3-ubyte"
trainLabels = "./MnistDataset/train-labels.idx1-ubyte"
testImages = "./MnistDataset/t10k-images.idx3-ubyte"
testLabels = "./MnistDataset/t10k-labels.idx1-ubyte"

classesNo = 10

images_train = readImages(trainImages) / 255.0
labels_train = readLabels(trainLabels)
images_test = readImages(testImages) / 255.0
labels_test = readLabels(testLabels)

y_train = keras.utils.to_categorical(labels_train, classesNo)
y_test = keras.utils.to_categorical(labels_test, classesNo)

images_train = images_train.reshape((images_train.shape[0],28,28,1))
images_test = images_test.reshape((images_test.shape[0],28,28,1))

resizedTrainImages = resizeData(images_train)
resizedTestImages = resizeData(images_test)

# hogTrainDescriptors = getHogDescriptors(resizedTrainImages,8,2)
# hogTestDescriptors = getHogDescriptors(resizedTestImages,8,2)
# hogTrainDescriptors16_4 = getHogDescriptors(resizedTrainImages,16,1)
# hogTestDescriptors16_4 = getHogDescriptors(resizedTestImages,16,1)

# file1 = open("hogTrainDescriptors.npy",'wb')
# file2 = open("hogTest,Descriptors.npy",'wb')
# file3 = open("hogTrainDescriptors16_4.npy","wb")
# file4 = open("hogTestDescriptors16_4.npy","wb")
# np.save(file3,hogTrainDescriptors16_4)
# np.save(file4,hogTestDescriptors16_4)
# np.save(file1,hogTrainDescriptors)
# np.save(file2,hogTestDescriptors)

with open("hogTrainDescriptors.npy",'rb') as f:
    hogTrainDescriptors = np.load(f)
with open("hogTestDescriptors.npy",'rb') as f:
    hogTestDescriptors = np.load(f)
with open("hogTrainDescriptors16_4.npy",'rb') as f:
    hogTrainDescriptors16_4 = np.load(f)
with open("hogTestDescriptors16_4.npy",'rb') as f:
    hogTestDescriptors16_4 = np.load(f)

linearKernelSVM16_4 = svm.SVC(kernel='linear', C=0.1)
linearKernelSVM16_4.fit(hogTrainDescriptors16_4,labels_train)

predictionsSVM16_4 = linearKernelSVM16_4.predict(hogTestDescriptors16_4)

confMatrixSVM16_4 = makeConfusionMatrix(labels_test, predictionsSVM16_4)
print(f'Accuracy:{metrics.accuracy_score(labels_test,predictionsSVM16_4)}')
print(confMatrixSVM16_4)

linearKernelSVM8_2 = svm.SVC(kernel='linear', C=0.1)
linearKernelSVM8_2.fit(hogTrainDescriptors,labels_train)

predictionsSVM8_2 = linearKernelSVM8_2.predict(hogTestDescriptors)

confMatrixSVM8_2 = makeConfusionMatrix(labels_test, predictionsSVM8_2)
print(f'Accuracy:{metrics.accuracy_score(labels_test,predictionsSVM8_2)}')
print(confMatrixSVM8_2)