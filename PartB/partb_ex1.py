
import numpy as np
import struct as st
import matplotlib.pyplot as plt
import tensorflow
import keras
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation,Input
from tensorflow.keras.models import Sequential
from PIL import Image
import cv2
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.lossFile = open("loss.txt","w")
        self.accFile = open("acc.txt","w")

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.lossFile.write(f'{loss}\n')
        self.accFile.write(f'{acc}\n')


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

def showNumbers(labels,images):
    for i in range(10):
        ind = np.where(labels == i)[0][0]
        plt.subplot(1,10,i+1)
        plt.imshow(images[ind,:,:])
    plt.show()   





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



# model = Sequential()
# model.add(Conv2D(6,kernel_size=(3,3),padding="same",strides=1,input_shape=(28,28,1)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2))
# model.add(Conv2D(16,kernel_size=(3,3),padding="same",strides=1))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2),strides=2))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Activation("relu"))
# model.add(Dense(84))
# model.add(Activation("relu"))
# model.add(Dense(10))
# model.add(Activation("softmax"))
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])


# model.fit(images_train, y_train,epochs=20, verbose=1, batch_size=8,callbacks=[TestCallback((images_test, y_test))])

# model = keras.models.load_model("./ModelCNN/")


loss = open("loss.txt","r")
acc = open("acc.txt","r")
loss = [float(x) for x in loss.read().splitlines()]
acc = [float(x) for x in acc.read().splitlines()]

# plt.subplot(1,2,1)
# plt.plot(loss)
# plt.subplot(1,2,2)
# plt.plot(acc)
# plt.show()


# predictions = model.predict(images_test).argmax(axis=1)

def makeConfusionMatrix(labels,predictions):
    confusionMatrix = np.zeros((10,10),dtype=np.int32)
    
    for i,j in zip(labels,predictions):
        confusionMatrix[i][j]+=1
    
    return confusionMatrix

# confMatrix = makeConfusionMatrix(labels_test,predictions)


from scipy.signal import convolve2d

def gradientMask(img):
    img = img.reshape((img.shape[0],img.shape[1]))
    Wx = np.array([-1,0,1]).reshape((3,1))
    Wy = Wx.T
    
    Gx = convolve2d(img,Wx,boundary="symm", mode="same")
    Gy = convolve2d(img,Wy,boundary="symm", mode="same")

    mag  = np.hypot(Gx,Gy)
    theta = np.arctan2(Gy,Gx)

    return mag,theta

def gradientData(data):
    
    n,r,c = data.shape
    images = data.reshape((n,r,c))
    magnitudes = np.zeros((n,r,c))
    angles = np.zeros((n,r,c))
    for i in range(n):
        mag,theta = gradientMask(images[i,:,:])
        magnitudes[i,:,:] = cv2.normalize(mag, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        angles[i,:,:] = cv2.normalize(theta, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    return magnitudes,angles

def resizeData(data):
    newSize = (64,128)
    n = data.shape[0]
    newData = data.reshape((n,28,28))
    resizedImages = np.zeros((n,newSize[1],newSize[0]))
    for i in range(n):
        img = Image.fromarray(newData[i, :, :]).resize(newSize)
        resizedImages[i, :, :] = np.asarray(img)
    
    return resizedImages
    
def getHogDescriptors(data):

    hogDescriptors = np.zeros((data.shape[0],7*15*36))

    for i in range(data.shape[0]):
        hogDescriptors[i,:] = hog(data[i,:], orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2),block_norm="L1")
    return hogDescriptors

# newTrainedImages = resizeData(images_train)
# mags, angles = gradientData(newTrainedImages)

# file1 = open("resizedTrainImages.npy","wb")
# file2 = open("magnitudes.npy","wb")
# file3 = open("angles.npy","wb")
# np.save(file1,newTrainedImages)
# np.save(file2,mags)
# np.save(file3,angles)


resizedTrainImages = resizeData(images_train)
resizedTestImages = resizeData(images_test)

# hogTrainDescriptors = getHogDescriptors(resizedTrainImages)
# hogTestDescriptors  = getHogDescriptors(resizedTestImages)

# file1 = open("hogTrainDescriptors.npy",'wb')
# file2 = open("hogTest,Descriptors.npy",'wb')
# np.save(file1,hogTrainDescriptors)
# np.save(file2,hogTestDescriptors)

with open("hogTrainDescriptors.npy",'rb') as f:
    hogTrainDescriptors = np.load(f)

print(hogTrainDescriptors.shape)

for i in hogTrainDescriptors[0,:]:
    print(i)