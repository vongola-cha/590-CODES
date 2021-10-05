

#CODE MODIFIED FROM:
# chollet-deep-learning-in-python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame

#------------------------
#EXPLORE IMAGE
#------------------------

#QUICK INFO ON IMAGE
def get_info(image):
	print("\n------------------------")
	print("INFO")
	print("------------------------")
	print("SHAPE:",image.shape)
	print("MIN:",image.min())
	print("MAX:",image.max())
	print("TYPE:",type(image))
	print("DTYPE:",image.dtype)

#SURFACE PLOT
def surface_plot(image):
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d') #viridis
    ax.plot_surface(xx, yy, image[:,:] ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
    plt.show()

#CHECK NUMBER OF EACH INSTANCE
def CheckCount(data):
	S=0
	for i in range(0,10):
		count=0
		for j in range(0,len(data)):
			if(data[j]==i):
				count+=1
		print("label =",i, "    count =",count)
		S+=count
	print("TOTAL =",S)

##GET DATASET
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

get_info(train_images)
get_info(train_labels)
get_info(test_images)
get_info(test_labels)
get_info(train_images[0])

print("\n----- TRAINING ------")
CheckCount(train_labels)
print("\n-----   TEST   ------")
CheckCount(test_labels)
plt.imshow(train_images[0], cmap=plt.cm.gray); plt.show()
surface_plot(train_images[0])

#SHOW FIRST IMAGE
image=train_images[0]

from skimage.transform import rescale, resize, downscale_local_mean
image = resize(image, (10, 10), anti_aliasing=True)
print((255*image).astype(int))
get_info(image)
plt.imshow(image, cmap=plt.cm.gray); plt.show()
exit()

print(image)
exit()
#plt.imshow(image)
plt.imshow(image, cmap=plt.cm.gray)
plt.show()

#PRETTY PRINT THE IMAGE MATRIX 
print(DataFrame(image))








