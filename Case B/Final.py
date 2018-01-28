from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
import numpy as np
import pickle
import sklearn.utils
import keras.backend as K
import keras.utils
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import time
import os

class CIFAR10Classifier(object):
    def __init__(self, batch_size=128, epochs=50, optimizer=keras.optimizers.Adam):
        self.model = None
        self.img_size = 32 # Width and height of each image.
        self.num_channels = 3 # Number of channels in each image, 3 channels: Red, Green, Blue.
        self.img_size_flat = self.img_size * self.img_size * self.num_channels # Length of an image when flattened to a 1-dim array.
        self.num_classes = 10 # Number of classes.
        self.num_files_train = 5 # Number of files for the training-set.
        self.images_per_file = 10000 # Number of images for each batch-file in the training-set.
        self.num_train_samples = self.num_files_train * self.images_per_file # Total number of images in the training-set. This is used to pre-allocate arrays for efficiency.
        self.input_shape = None

        # Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.opt = optimizer()

    def load_data(self):
        x_train = np.empty((self.num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((self.num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = 'cifar10/data_batch_' + str(i)
            (x_train[(i - 1) * 10000: i * 10000, :, :, :],
             y_train[(i - 1) * 10000: i * 10000]) = self._load_batch(fpath)

        fpath = 'cifar10/valid_batch'
        x_test, y_test = self._load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test  = np.reshape(y_test, (len(y_test), 1))
        
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
        
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        # Normalize and convert to float32
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # Shuffle x_train for better model
        x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=0)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = self.x_train.shape[1:]

    def _load_batch(self, fpath, label_key='labels'):       
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
        data = d['data']
        labels = d[label_key]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    def make_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation('softmax'))

    def fit_model(self):
        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.opt,
                      metrics=['accuracy'])

        self.history = self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1)
        self.history = self.history.history
        self.model = self.model

    def visualize(self):
        # summarize history for accuracy
        plt.plot(self.history['acc'])
        plt.plot(self.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def test_model(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print("Loss (Lower is better): %.2f" % loss)
        print('Accuracy (Higher is better): %.2f' % acc)

    def load_model(self):
        try:
            self.model = keras.models.load_model("model.hdf5")
            self.history = joblib.load('history.pckl')
        except:
            self.model = None
            self.history = None

    def save_model(self):
        self.model.save("model.hdf5")
        joblib.dump(self.history, 'history.pckl') 

def main():
    relearn = False
    cifar10model = CIFAR10Classifier(epochs=100)
    if (os.path.exists("model.hdf5") and os.path.exists("history.pckl") and relearn == False):
        cifar10model.load_model()
        cifar10model.visualize()
    else:
        cifar10model.load_data()
        cifar10model.make_model()
        cifar10model.fit_model()
        cifar10model.test_model()
        cifar10model.save_model()

if __name__ == '__main__':
    main()