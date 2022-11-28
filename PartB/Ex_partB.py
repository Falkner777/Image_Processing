

import numpy as np
import struct as st
import matplotlib.pyplot as plt
import tensorflow
import keras
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation,Input,AveragePooling2D
from tensorflow.keras.models import Sequential

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
        plt.imshow(images[ind,:,:],cmap='gray')
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
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



model = Sequential()
model.add(Conv2D(6,kernel_size=(3,3),padding="same",strides=1,input_shape=(28,28,1)))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(16,kernel_size=(3,3),padding="same",strides=1))
model.add(Activation("relu"))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(120))
model.add(Activation("relu"))
model.add(Dense(84))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(images_train, y_train,epochs=50, verbose=1, batch_size=50,callbacks=[TestCallback((images_test, y_test))])

keras.models.save_model(model,"./ModelCNN/")
showNumbers(labels_train,images_train)


loss = open("loss.txt","r")
acc = open("acc.txt","r")
loss = [float(x) for x in loss.read().splitlines()]
acc = [float(x) for x in acc.read().splitlines()]

plt.subplot(1,2,1)
plt.plot(loss,"-")
plt.title("Loss")
plt.subplot(1,2,2)
plt.plot(acc,"-")
plt.title("Accuracy")
plt.show()


predictions = model.predict(images_test).argmax(axis=1)

def makeConfusionMatrix(labels,predictions):
    confusionMatrix = np.zeros((10,10),dtype=np.int32)
    
    for i,j in zip(labels,predictions):
        confusionMatrix[i][j]+=1
    
    return confusionMatrix

confMatrix = makeConfusionMatrix(labels_test,predictions)
print(confMatrix)
