import numpy as np
from keras import layers 
from keras import models

#USER PARAM 
N=5;		#NUMBER OF PIXELS IN X AND Y
KS=2 		#KERNAL SIZE
NK=2 		#NUMBER OF KERNALS 

#MAKE TOY RGB IMAGE
R=np.ones(N*N).reshape(N,N,1)
G=np.ones(N*N).reshape(N,N,1)+1
B=np.ones(N*N).reshape(N,N,1)+2
X=np.concatenate((R,G,B), axis=2).reshape(1,N,N,3)
print(X[0,:,:,1]); print("X SHAPE:",X.shape)

#DEFINE KERAS MODEL 
model = models.Sequential()
conv2d =layers.Conv2D(
	filters=NK, 
	kernel_size=KS, 
	activation=None,
	padding='valid', 
	kernel_initializer='ones', 
	bias_initializer='zeros',
	input_shape=(N, N, 3)
	)


model.add(conv2d)
# model.add(layers.MaxPooling2D((2, 2)))

# # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# # model.add(layers.MaxPooling2D((2, 2)))

# # model.add(layers.Flatten())
# # model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(10, activation='softmax'))

model.summary()

def explore_keras_model(model):
	print("INPUT SHAPE:",model.inputs[0].shape)
	print("N LAYERS:",len(model.layers))
	for layer in model.layers:
		print("----------LAYER----------")
		print("  NAME:", layer.name)
		print("  TRAINABLE", layer.trainable)
		print("  INPUT",layer.input_spec)
		# print("  TRAINABLE WEIGHTS",layer.trainable_weights)
		if(len(layer.trainable_weights)>0):
			print("  TRAINABLE WEIGHTS SHAPE:",(layer.trainable_weights[0]).shape)
			tmp=layer.get_weights(); NFIT=0
			for j in range(0,len(tmp)):
				array=tmp[j]; NFIT+=array.size
				if(array.ndim==4 and j==0): #PRINT FIRST KERNAL
					print("  KERNAL:")
					print(array[:,:,0,0])
				if(array.ndim==1):
					print("  BIAS:",array)
				print("  WEIGHT SHAPE/SIZE",j,array.shape,array.size)
			print("  NFIT",NFIT)

# print("NFIT=", X.shape[3]*NK*KS*KS+NK)
explore_keras_model(model)

#MAKE PERDICTION
Y=model.predict(X)
print(Y)
print(Y.shape)

exit()




# #APPLY FILTER
# def convolution(X,k,pad='valid'):
# 	# X=image MUST BE NxMxC (c=channels)
# 	# k=kernal

# 	if(X.ndim!=3): 
# 		print("ERROR"); exit()
# 	tmp=np.copy(X)
# 	if(pad=='same'):
# 		print("NO CODED"); exit()

# 	if(pad=='valid'):
# 		for channel in range(0,x.shape[2]):
# 			for i in range(1,x.shape[0]-1):
# 				for j in range(1,x.shape[1]-1):
# 					sub=x[:,:,channel]
# 					sub_matrix=sub[i-1:i+2,j-1:j+2] 
# 					tmp[i,j,channel]=np.sum(filter_ED*sub_matrix)

#     # tmp=tmp.astype(int)
# 	return tmp

