
import tensorflow as tf
import numpy as np


#SELECT EXAMPLE TO RUN 
EXAMPLE=1
N_TIMESTEPS=10

def report(x,y,l):
	print("X:"); print(x); 
	print("SHAPE:",x.shape)
	print("KERNAL SHAPE:",l.get_weights()[0].shape)
	print("KERNAL:"); print(l.get_weights()[0])
	print("Y SHAPE",y.shape)
	print("Y:"); print(y)

#-------------------------------
#EXAMPLE-1: 1 FEATURE, 1 SAMPLE
#-------------------------------
if(EXAMPLE==1):
	N_SAMPLES=1 #MUST EQUAL 1
	N_FEATURES=1
	x=np.linspace(N_SAMPLES,N_TIMESTEPS,N_TIMESTEPS)
	x=x.reshape(N_SAMPLES,N_TIMESTEPS,N_FEATURES)
	layer1 = tf.keras.layers.Conv1D(
		1, 2, activation='relu',
	    kernel_initializer="ones",
	    bias_initializer="zeros",
		input_shape=x.shape[1:]
		)
	y = layer1(x)
	report(x,y,layer1)

#-------------------------------
#EXAMPLE-2: N FEATURES, 1 SAMPLE
#-------------------------------
if(EXAMPLE==2):
	N_SAMPLES=1 #MUST EQUAL 1
	N_FEATURES=2
	x=np.zeros((N_TIMESTEPS,N_FEATURES))
	for i in range(0,N_FEATURES):
		x[:,i]=(i+1)*np.linspace(1,N_TIMESTEPS,N_TIMESTEPS)
	x=x.reshape(N_SAMPLES,N_TIMESTEPS,N_FEATURES)
	layer1=tf.keras.layers.Conv1D(
		1, 2, activation='relu',
	    kernel_initializer="ones",
	    bias_initializer="zeros",
		input_shape=x.shape[1:]
		)
	y = layer1(x)
	report(x,y,layer1)


#-------------------------------
#EXAMPLE-3: N FEATURES, M SAMPLE
#-------------------------------
if(EXAMPLE==3):
	N_SAMPLES=2
	N_FEATURES=2
	x=np.zeros((N_SAMPLES,N_TIMESTEPS,N_FEATURES))
	for j in range(0,N_SAMPLES):
		for i in range(0,N_FEATURES):
			x[j,:,i]=(j+5)*(i+1)*np.linspace(1,N_TIMESTEPS,N_TIMESTEPS)
	x=x.reshape(N_SAMPLES,N_TIMESTEPS,N_FEATURES)
	layer1=tf.keras.layers.Conv1D(
		1, 2, activation='relu',
	    kernel_initializer="ones",
	    bias_initializer="zeros",
		input_shape=x.shape[1:]
		)
	y = layer1(x)
	report(x,y,layer1)


#GET LAYER INFO
from keras.models import Sequential 
model= Sequential()
model.add(layer1)
model.summary()



# N_FEATURES=2
# N_TIMESTEPS=100
# N_SAMPLES=1
# NFILTERS=32
# KERNAL_SIZE=3

# in_shape = (N_SAMPLES, N_TIMESTEPS, N_FEATURES)

# #GENERATE RANDOM DATA
# x = tf.random.normal(in_shape)
# print(x.shape)
# y = tf.keras.layers.Conv1D(
# 	32, 3, activation='relu',input_shape=in_shape[1:]
# 	)(x)
# print(in_shape[1:])
# print(y.shape)
