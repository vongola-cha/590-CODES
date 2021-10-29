


import os 
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image


ISHOW=False
NX=200 #new resolution
NY=NX



#TRAINING
def clean(name='train'):
	OUT=name+"-clean"
	if(os.path.exists(OUT)): shutil.rmtree(OUT) 
	os.mkdir(OUT)

	path1='HW5.0-DATASET/'+name+'/'
	list_1= os.listdir(path1)

	counter=1
	for folder in list_1:
		print(folder)
		path2=path1+folder
		print(len(os.listdir(path2)))
		# for file in os.listdir(path2):

		# 	#LOAD IMAGE
		# 	x=plt.imread(path2+'/'+file)

		# 	#RESIZE
		# 	from skimage.transform import rescale, resize, downscale_local_mean
		# 	x = 255*resize(x, (NY, NX), anti_aliasing=True)
		# 	print(counter,x.shape)
		# 	x=x.astype(np.uint8)

		# 	#SAVE
		# 	im = Image.fromarray(x)

		# 	#SAVE ORIGINAL 
		# 	im.save(OUT+'/'+str(folder)+'-'+str(counter)+'.jpg')

		# 	# #-------------------------------
		# 	# #DATA AUGMENT
		# 	# #--------------------------------
		# 	# if(name=='train'):

		# 	# 	#ROTATIONS
		# 	# 	for angle in [45,90,135,180,225,270,315]:
		# 	# 		augmented = im.rotate(angle)
		# 	# 		augmented.save(OUT+'/'+str(folder)+'-'+str(counter)+'-angle-'+str(angle)+'.jpg')

		# 	counter+=1

		# 	if(ISHOW): 
		# 		plt.imshow(x); #plt.show()
		# 		plt.show(block=False)
		# 		plt.pause(0.3)




#RUN
clean('train')
clean('test')