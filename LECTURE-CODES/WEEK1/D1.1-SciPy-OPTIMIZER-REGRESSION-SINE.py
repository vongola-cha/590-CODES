#--------------------------------
#SIMPLE UNIVARIABLE REGRESSION EXAMPLE
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from   scipy.optimize import curve_fit

#------------------------
#GENERATE TOY DATA
#------------------------

#GROUND-TRUTH PARENT FUNCTION
def f(x):  #vectorized; array [x1,x2 ... xN] --> f [y1,y2 ... yN]
	out=np.sin(x)
	# out=x
	# out=x**2.0
	#out=np.exp(-x**2.0)
	return out

#PARAMETERIZED GROUND-TRUTH 
def G(x,p=[1.,1.,1,1.]):
	return p[0]*f((x-p[1])/p[2])+p[3]

#NOISY DATA
N=1000; xmin=-10; xmax=10
x = np.linspace(xmin,xmax,N)
ground_truth_param=np.random.uniform(0,1,size=4)
print("ground_truth_param",ground_truth_param)
y = G(x,ground_truth_param)  #PRISTINE DATA
#noise=np.random.normal(loc=0.0, scale=0.05*(max(y)-min(y)),size=len(x))
noise=0.15*(max(y)-min(y))*np.random.uniform(-1,1,size=len(x))
yn = y + noise

#GROUND TRUTH DATA
xe = np.linspace(xmin,2*xmax,int(2*N))
ye = G(xe,ground_truth_param)

#------------------------
#FIT MODEL
#------------------------

# ##FITTING MODEL
model_type="L";  po=[1.,1.]		#linear
model_type="Q";  po=[1.,1.,1.0]		#quad
model_type="S";  po=np.random.uniform(0.,1.,size=4)

def model(x,p):
	if(model_type=="L"): return p[0]*x+p[1] 
	if(model_type=="Q"): return p[0]*x*x+p[0]*x+p[1] 
	if(model_type=="S"): return p[0]*np.sin((x-p[1])/p[2])+p[3]

count=0
def loss(p):
	global count
	yp=model(x,p) #model predictions for given parameterization p
	#print(yp,yn)
	RMSE=(np.mean((yn-yp)**2.0))*0.5
	count+=1
	if(count%25==0):
		print(count,RMSE,p)
	return RMSE

from scipy.optimize import minimize
res = minimize(loss, po, method='BFGS', tol=1e-15)
popt=res.x
print("OPTIMAL PARAM:",popt)

#------------------------
#PLOT
#------------------------

fig, ax = plt.subplots()
ax.plot(x, yn, 'o', label='Data')
ax.plot(xe, ye, '-', label='Ground-Truth')
ax.plot(xe, model(xe, popt), 'r-', label="Model")

ax.legend()
FS=18   #FONT SIZE
plt.xlabel('Time (s) ', fontsize=FS)
plt.ylabel('Spring displacement (cm)', fontsize=FS)

plt.show()

