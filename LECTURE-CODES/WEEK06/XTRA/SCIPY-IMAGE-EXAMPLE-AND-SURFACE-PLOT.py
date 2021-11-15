#DEPENDENCY: #conda install scikit-image

#CODE MODIFIED FROM:
# https://stackoverflow.com/questions/31805560/how-to-create-surface-plot-from-greyscale-image-with-matplotlib
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.face.html#scipy.misc.face

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#LOAD IMAGE
image=plt.imread('luna-1.jpeg')

#------------------------
#EXPLORE IMAGE
#------------------------

# #SHOW ORIGINAL IMAGE
plt.imshow(image); plt.show()

# #ROTATE BY SWITCHING X AND Y DIMENSIONS
# image=np.transpose(image,axes=[1,0,2])
# plt.imshow(image); plt.show()


#QUICK INFO ON IMAGE
def get_info(image):
	print("------------------------")
	print("IMAGE INFO")
	print("------------------------")
	print(type(image))
	print(image.shape)
	print("NUMBER OF PIXELS=",image.shape[0]*image.shape[1])
	print("NUMBER OF MATRIX ENTRIES=",image.size)
	# print("N CHANNELS",image.shape[2])
	print(image.min())
	print(image.max())
	print(image.dtype)
	print("pixel-1 :", image[0,0])
	# print("image[0:3].shape:", image[0:3].shape)


#GET ORIGINAL INFO
get_info(image)

#SLICE IMAGE
# plt.imshow(image[500:2000]); plt.show()
# plt.imshow(image[:,1000:2000]); plt.show()

#REDUCE RESOLUTION-1
from skimage.transform import rescale, resize, downscale_local_mean

factor=50
# image = resize(image, (image.shape[0] // factor, image.shape[1] // factor), anti_aliasing=True)

image = resize(image, (10, 10), anti_aliasing=True)
get_info(image)
plt.imshow(image); plt.show()
exit()

#CONVERT TO GRAYSCALE
from skimage.color import rgb2gray 
image=rgb2gray(image)
plt.imshow(image, cmap=plt.cm.gray)
plt.show()
get_info(image)

#SURFACE PLOT
def surface_plot(image):
	# create the x and y coordinate arrays (here we just use pixel indices)
	xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
	fig = plt.figure()
	ax = fig.gca(projection='3d') #viridis
	ax.plot_surface(xx, yy, image[:,:] ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
	plt.show()

surface_plot(image)
