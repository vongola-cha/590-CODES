
import numpy as np
import matplotlib.pyplot as plt

#GENERATE DATA
N=100; f=2; A=10
t=np.linspace(0,10/f,N)
y=A*np.sin(2*3.14*t/f)+np.random.uniform(-A/3,A/3,N)

#DEFINE KERNAL: 
w=6			#AVERAGING WINDOW (3 --> +1 0 -1)

mask=np.ones((1,w))/w; mask=mask[0,:] 
# mask=np.random.uniform(-1,1,w)
mask=np.random.normal(0,A/10,w)
# mask=np.array([1,0,0,-2,0,0,-1])/2.0
print(mask)

#Convolve the mask with the raw data
yc=np.convolve(y,mask,'same')
 
#PLOT  DATA 
fig, ax = plt.subplots()
ax.plot(t,y,'o-',t,yc,'.-')
plt.show()

exit()
