

from scipy.integrate import quad
import numpy as np

import matplotlib.pyplot as plt


#-----------------------------------
#INTEGRATION EXAMPLE
#----------------------------------

#ANTIDEVIATIVE OF x^2--> x^3/3
def integrand(x):
    return x**2 

print(1/3-0,quad(integrand, 0, 1))

#-----------------------------------
#NORMAL DISTIBUTION ENTROPY
#----------------------------------

#LOOP OVER VARIOUS SIGMA AND PLOT 
ENTROPY=[]; 
SIGMA=[0.1,0.5,1,2,3,4,5,6,7,8,9,10]
for s in SIGMA:
	u=0
	def p(x):
		return np.exp(-0.5*((x-u)/s)**2)/(s*np.sqrt(2*3.1415))

	def I(x): return -p(x)*np.log(p(x))

	S=quad(I, -5*s, 5*s)
	print(S[0], np.log(((2*np.pi*np.exp(1))**0.5)*s)) #,np.exp(1))

	ENTROPY.append(S[0])

	#PLOT DISTRIBUTIONS
# 	x=np.linspace(-3*s,3*s,500)
# 	plt.plot(x,p(x),'-')

# plt.plot(x,0*x/x,'-')
# plt.show()


#PLOT
s=np.linspace(min(SIGMA),max(SIGMA),500)
fig = plt.figure(figsize=(20,12))
ax  = fig.add_subplot(111)
plt.plot(SIGMA,ENTROPY,'bo')
plt.plot(s, np.log(((2*np.pi*np.exp(1))**0.5)*s),'-')
ax.set_xlabel('standard deviation')
ax.set_ylabel('entropy of system')
plt.show()
