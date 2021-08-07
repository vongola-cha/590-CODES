

#CODE MODIFIED FROM:
# chollet-deep-learning-in-python

import numpy as np
from pandas import DataFrame

#------------------------
#EXPLORE IMAGE
#------------------------

#QUICK INFO ON NP ARRAY
def get_info(X):
	print("\n------------------------")
	print("SUMMARY")
	print("------------------------")
	print("TYPE:",type(X))

	if(str(type(x))=="<class 'numpy.ndarray'>"):

		print("SHAPE:",X.shape)
		print("MIN:",X.min())
		print("MAX:",X.max())
		#NOTE: ADD SLICEING
		print("DTYPE:",X.dtype)
		print("NDIM:",X.ndim)
		#PRETTY PRINT 
		if(X.ndim==1 or X.ndim==2 ): 
			print("EDGES ARE INDICES: i=row,j=col") 
			print(DataFrame(X)) 	
	else:
		print("ERROR: INPUT IS NOT A NUMPY ARRAY")

#SCALAR (0D TENSOR)
x = np.array(10); get_info(x)

#VECTOR AS 1D ARRARY
x = np.array([12, 3, 6, 14]); get_info(x)

#VECTOR AS 2D ARRAY 
x = np.array([12, 3, 6, 14]);  x=x.reshape(len(x),1); get_info(x) #COLUMN VECTOR
x = np.array([12, 3., 6, 14]); x=x.reshape(1,len(x)); get_info(x) #ROW VECTOR

#MATRIX (2D TENSOR)
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]); get_info(x)

#3D TENSOR
x = np.array([[[5., 78, 2, 34, 0],
			   [6, 79, 3, 35, 1],
			   [7, 80, 4, 36, 2]],
			  [[5, 78, 2, 34, 0],
			   [6, 79, 3, 35, 1],
			   [7, 80, 4, 36, 2]],
			  [[5, 78, 2, 34, 0],
			   [6, 79, 3, 35, 1],
			   [7, 80, 4, 36, 2]]]) ; get_info(x)
