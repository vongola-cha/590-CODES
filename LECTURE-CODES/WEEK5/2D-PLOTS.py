

##-------------------------------------------
## 2 VARIABLE NORMAL DISTIBUTION
##-------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

FUNC=2
FS=18	#FONT SIZE

u=np.array([[0.0],[0.0]]) #u=[ux,uy] 
s=np.array([[0.5,0.0],[0.0,1.0]])


# DEFINE FUNCTION 
def N(x, y):
	ux=u[0,0]; uy=u[1,0]
	sx=s[0,0]; sy=s[1,1]; p=s[0,1]
	# print(ux,uy,sx,sy,p)
	out=1.0/(2*3.1415*sx*sy*(1-p**2.0)**0.5)
	out=out*np.exp(-(((x-ux)/sx)**2.0-2*p*((x-ux)/sx)*((y-uy)/sy)+((y-uy)/sy)**2.0)/(2*(1-p**2)))
	return out


#MESH-1 (SMALLER)
xmin=-3; xmax=3; ymin=xmin; ymax=xmax
x,y = np.meshgrid(np.linspace(xmin,xmax,20),np.linspace(ymin,ymax,20))

#MESH-2 (DENSER)
X, Y = np.meshgrid(np.linspace(xmin, xmax, 40), np.linspace(ymin, ymax, 40))

#CONTOUR PLOT 
CMAP='hsv' #'RdYlBu'
plt.contour(X, Y, N(X, Y), 20, cmap=CMAP); 
plt.show(); 

#SURFACE PLOT 
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('x', fontsize=FS); ax.set_ylabel('y', fontsize=FS); ax.set_zlabel('p(x,y)', fontsize=FS)
surf=ax.plot_surface(X, Y, N(X, Y), cmap=CMAP) 
plt.plot(0,0, 0.3, '.', markersize=0.1) 
plt.show(); 








# exit()
# #DEFINE FUNCTION (MATRIX FORM)
# def N(x, y):

# 	X=np.array([[x],[y]])
# 	#print(u.shape,s.shape)
# 	# X=
# 	# print(x.shape,y.shape); exit()
# 	det=np.linalg.det(s)
# 	inv=np.linalg.inv(s)
# 	# print(np.transpose(X-u).shape,X.shape,u.shape); exit()
# 	out=-0.5*np.matmul(np.transpose(X-u),inv)
# 	out=np.matmul(out,X-u)
# 	out=np.exp(out)/np.sqrt(((2*np.pi)**2.0)*det)
# 	out=out[0][0]
# 	#print(det,out)

# 	out=1
# 	return out
# N(1,1)
# exit()