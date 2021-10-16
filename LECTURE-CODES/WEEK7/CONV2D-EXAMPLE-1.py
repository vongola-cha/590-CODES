
#-------------------------------------------
#EXAMPLE-2
#-------------------------------------------

import numpy as np
from keras.layers import Input, Conv2D
from keras.models import Model

#FORM IMAGE
red   = np.array([1]*9).reshape((3,3))
green = np.array([100]*9).reshape((3,3))
blue  = np.array([10000]*9).reshape((3,3))
img = np.stack([red, green, blue], axis=-1)
img = np.expand_dims(img, axis=0)

#CONSTRUCT MODEL 
inputs = Input((3,3,3))
conv = Conv2D(filters=1, 
              strides=1, 
              padding='valid', 
              activation='relu',
              kernel_size=2, 
              kernel_initializer='ones', 
              bias_initializer='zeros', )(inputs)
model = Model(inputs,conv)
model.summary()

#EVALUATE ON IMAGE
Y=model.predict(img)
print(Y)
print(Y.shape)

