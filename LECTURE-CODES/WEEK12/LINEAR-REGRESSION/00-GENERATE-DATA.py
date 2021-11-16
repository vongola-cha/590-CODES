

import  numpy           as  np
import  matplotlib.pyplot   as  plt

#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------


#DATA PARAM
a       =   1.0; 
b       =   1.0; 
max_noise   =   0.35

#GROUND TRUTH       
def F1(x, A, B):
    return A*x+B

#MESH PARAM
xo      =   -4. #0.75; 
xf      =   +4. #0.75;  
N       =   100;
XMID    =   (xf+xo)/2.0
DX      =   (xf-xo)

#DEFINE VARIOUS ARRAYS
X1      =   np.linspace(xo,xf,N).reshape(N,1)
YGT1    =   F1(X1, a, b)  #GROUND TRUTH
NOISE   =   2*np.random.uniform(-max_noise,max_noise,N).reshape(N,1)
YN      =   YGT1+NOISE      #NOISY DATA

#CONCAT
OUT=np.append(X1,YN, axis=1)

print(OUT.shape)

#PLOT DATA SET
plt.figure()
plt.plot(OUT[:,0], OUT[:,1], 'o', label="Training data")
plt.legend()
plt.show()

#SAVE
np.save('X-Y',OUT)