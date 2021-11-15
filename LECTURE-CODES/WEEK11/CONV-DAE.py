

#SOURCE: MODIFIED FROM https://blog.keras.io/building-autoencoders-in-keras.html

import keras
from keras import layers
import matplotlib.pyplot as plt

from keras.datasets import mnist,cifar10
import numpy as np

#USER PARAM
INJECT_NOISE    =   False
EPOCHS          =   35
NKEEP           =   2500        #DOWNSIZE DATASET
BATCH_SIZE      =   128
DATA            =   "MNIST"

#GET DATA
if(DATA=="MNIST"):
    (x_train, _), (x_test, _) = mnist.load_data()
    N_channels=1; PIX=28

if(DATA=="CIFAR"):
    (x_train, _), (x_test, _) = cifar10.load_data()
    N_channels=3; PIX=32
    EPOCHS=250 #OVERWRITE

#NORMALIZE AND RESHAPE
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#DOWNSIZE TO RUN FASTER AND DEBUG
print("BEFORE",x_train.shape)
x_train=x_train[0:NKEEP]
x_test=x_test[0:NKEEP]
print("AFTER",x_train.shape)

#ADD NOISE IF DENOISING
if(INJECT_NOISE):
    EPOCHS=2*EPOCHS
    #GENERATE NOISE
    noise_factor = 0.5
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_train=x_train+noise
    noise= noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
    x_test=x_test+noise

    #CLIP ANY PIXELS OUTSIDE 0-1 RANGE
    x_train = np.clip(x_train, 0., 1.)
    x_test = np.clip(x_test, 0., 1.)

#BUILD CNN-AE MODEL
input_img = keras.Input(shape=(PIX, PIX, N_channels))

#ENCODER
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# AT THIS POINT THE REPRESENTATION IS (4, 4, 8) I.E. 128-DIMENSIONAL

#DECODER
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)

#EITHER PAD OR NOT TO MAKE OUTPUT SHAPE CORRECT
if(DATA=="MNIST"):
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
if(DATA=="CIFAR"):
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)

#OUTPUT
decoded = layers.Conv2D(N_channels, (3, 3), activation='sigmoid', padding='same')(x)

#COMPILE
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy');
autoencoder.summary()

#TRAIN
history = autoencoder.fit(x_train, x_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                validation_data=(x_test, x_test),
                )

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()

#MAKE PREDICTIONS FOR TEST DATA
decoded_imgs = autoencoder.predict(x_test)

#VISUALIZE THE RESULTS
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(PIX, PIX,N_channels))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

