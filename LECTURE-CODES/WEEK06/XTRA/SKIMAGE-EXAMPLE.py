#------------------------------------------
#BASIC IMAGE MANIPULATION 
#------------------------------------------

from skimage import io
import matplotlib.pyplot as plt
import numpy as np

#The different color bands/channels are stored in the third dimension, 
#such that a gray-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
image = io.imread('./luna-1.jpeg')

#SHOW ORIGINAL IMAGE
plt.imshow(image); plt.show()

N_PIXEL=image.shape[0]*image.shape[1]
pixel1=image[0,0,:]
print("SHAPE: ", image.shape)
print("PIXEL=1: ", pixel1)
print("NUMBER OF PIXELS:",N_PIXEL)

#ZOOM IN ON UPPER REGION
tmp=image[0:400,0:400,:]; plt.imshow(tmp); plt.show()
#BLACK OUT REGIONS (MAKE PIXELS=0,0,0)
tmp=np.copy(image); tmp[100:200,:,:]=0*pixel1; plt.imshow(tmp); plt.show()
tmp=np.copy(image); tmp[:,100:200,:]=0*pixel1; plt.imshow(tmp); plt.show()

#CONVERT TO GRAYSCALE
from skimage.color import rgb2gray
image = rgb2gray(image)
plt.imshow(image, cmap=plt.cm.gray); plt.show()

#SWIRL
from skimage.transform import swirl
swirled = swirl(image, rotation=0, strength=10, radius=500)

plt.imshow(swirled, cmap=plt.cm.gray); plt.show()

#------------------------------------------
#EDGE DETECTION
#------------------------------------------
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)
edge_scharr = filters.scharr(image)
edge_prewitt = filters.prewitt(image)

fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_roberts, cmap=plt.cm.gray)
axes[0].set_title('Roberts Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

axes[2].imshow(edge_scharr, cmap=plt.cm.gray)
axes[2].set_title('Scharr Edge Detection')

axes[3].imshow(edge_prewitt, cmap=plt.cm.gray)
axes[3].set_title('prewitt Edge Detection')


for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()