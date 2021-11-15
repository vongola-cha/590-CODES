

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

s=1; u=0
def p(x):
	return np.exp(-0.5*((x-u)/s)**2)/(s*np.sqrt(2*3.1415))

def I(x): return -p(x)*np.log(p(x))

S=quad(I, -20, 20)
print(S, np.log(2*3.1415*np.exp(1)*s)) #,np.exp(1))


#-----------------------------------
#KL DIVERGENCE AND CROSS ENTROPY
#----------------------------------

u1=0; s1=1
u2=3; s2=1

#INTEGRATION LIMITS
x1=min(u1,u2)-4*max(s1,s2)
x2=max(u1,u2)+4*max(s1,s2); print(x1,x2)

#DEFINE TWO NORMAL DISTRIBUTINO
def p1(x): return np.exp(-0.5*((x-u1)/s1)**2)/(s1*np.sqrt(2*3.1415))
def p2(x): return np.exp(-0.5*((x-u2)/s2)**2)/(s2*np.sqrt(2*3.1415))

def I_S1(x): return -p1(x)*np.log(p1(x))  		#P1 ENTROPY
def I_S2(x): return -p2(x)*np.log(p2(x))  		#P2 ENTROPY
def I_CE(x): return -p1(x)*np.log(p2(x))  		#CROSS ENTROPY
def I_KL(x): return -p1(x)*np.log(p2(x)/p1(x))  #D_KL

def plot(KLD,S1,S2,CE):
	fig = plt.figure(figsize=(20,12))
	ax = fig.add_subplot(111)
	x=np.linspace(x1,x2,500)
	plt.plot(x, p1(x), label="p(x)", linewidth=4)
	plt.plot(x, p2(x), label="q(x)", linewidth=4)
	plt.plot(x, I_S1(x), label="I_S1", linewidth=4)
	# plt.plot(x, I_S2(x), label="I_S2", linewidth=4)
	plt.plot(x, I_CE(x), label="I_CE", linewidth=4)
	plt.plot(x, I_KL(x), label="S1,S2,CE,KLD="+str(S1)+' '+str(S2)+' '+str(CE)+' '+str(KLD), linewidth=4)
	plt.legend(loc="best")
	ax.set_xlabel('x')
	ax.set_ylabel('probablity density function')
	plt.show()

S1=round(quad(I_S1, x1, x2)[0],3)
S2=round(quad(I_S1, x1, x2)[0],3)
KLD=round(quad(I_KL, x1, x2)[0],3)
CE=round(quad(I_CE, x1, x2)[0],3)
print(CE,S1+KLD)
plot(KLD,S1,S2,CE)















exit()

#-----------------------------------
#KL DIVERGENCE: MOVIE 
#----------------------------------

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)

for var in [15,1.42,3,4,5,6,7,8,9]:
	u1=0; s1=1
	u2=0; s2=var

	#INTEGRATION LIMITS
	x1=min(u1,u2)-4*max(s1,s2)
	x2=max(u1,u2)+4*max(s1,s2); print(x1,x2)

	#DEFINE TWO NORMAL DISTRIBUTINO
	def p1(x): return np.exp(-0.5*((x-u1)/s1)**2)/(s1*np.sqrt(2*3.1415))
	def p2(x): return np.exp(-0.5*((x-u2)/s2)**2)/(s2*np.sqrt(2*3.1415))

	#INTEGRAND
	def I(x): return -p1(x)*np.log(p2(x)/p1(x))

	def plot(KLD):

		x=np.linspace(x1,x2,500)
		ax.clear()

		ax.plot(x, p1(x), label="p(x)", linewidth=4)
		ax.plot(x, p2(x), label="q(x)", linewidth=4)
		ax.plot(x, I(x), label="p(x)log(p(x)/q(x)): KLD="+str(KLD), linewidth=4)
		ax.legend(loc="best")
		ax.set_xlabel('x')
		ax.set_ylabel('probablity density function')
		plt.pause(1)


	KLD=quad(I, x1, x2)[0]
	plot(KLD)

plt.show()