
import json
import numpy as np
#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------


def write_json(x,name):
	with open(name, "w") as write_file:
		json.dump(x, write_file)

#--------------------
#planar_x1_x2_y.json
#--------------------

#GENERATE DATA
N=50
X1=[]; X2=[]; Y=[]
for x1 in np.linspace(-5,5,N):
	for x2 in np.linspace(-6,6,N):
		# noise=1*np.random.uniform(-1,1,size=1)[0]
		y=2.718*x1+3.14*x2+1.0 #+noise
		X1.append(x1)
		X2.append(x2)
		Y.append(y)
		print(x1,x2,y)

out={}
out['x1']=X1
out['x2']=X2
out['y']=Y
write_json(out,'planar_x1_x2_y.json')


#GENERATE DATA
N=25
X1=[]; X2=[]; X3=[]; Y=[]
for x1 in np.linspace(-5,5,N):
	for x2 in np.linspace(-6,6,N):
		for x3 in np.linspace(-10,10,N):
			# noise=1*np.random.uniform(-1,1,size=1)[0]
			y=2.718*x1+3.14*x2+1.4142*x3+1.0 #+noise
			X1.append(x1)
			X2.append(x2)
			X3.append(x3)
			Y.append(y)
			print(x1,x2,x3,y)

out={}
out['x1']=X1
out['x2']=X2
out['x3']=X3
out['y']=Y
write_json(out,'planar_x1_x2_x3_y.json')

exit()

#MISC PARAM
iplot		=	True

name="housing_price";   out={}
name="weight"; 			out={}


#GROUND-TRUTH PARENT FUNCTION
def f(x,name):  #vectorized; array [x1,x2 ... xN] --> f [y1,y2 ... yN]
	if(name=="housing_price"):
		out=500000*np.exp(-(x/3.)**2.0)+250000*np.exp(-((x-10.)/2.5)**2.0)+150000*np.exp(-((x+12)/3)**2.0)
	if(name=="weight"):
		out=181.0/(1+np.exp(-(x-13)/4))+20
	return out



if(name=="housing_price"):  
	out["xlabel"]="distance_miles"; out["ylabel"]="house_price"
	N=250; xmin=-20; xmax=20; SF=0.075

if(name=="weight"):  
	out["xlabel"]="age"; 
	out["ylabel"]="weight"
	N=250; xmin=3; xmax=100; SF=0.12



#GENERATE DATA
x = np.linspace(xmin,xmax,N)
y =f(x,name)  #PRISTINE DATA
noise=SF*(max(y)-min(y))*np.random.uniform(-1,1,size=len(x))
yn = y + noise

if(name=="weight"): 
	A_or_C=[] 
	for i in range(0,len(x)):
		if(x[i]<18):
			A_or_C.append(0)
		else:
			A_or_C.append(1)
	out["is_adult"]=A_or_C


if(iplot):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.plot(x, yn, 'o', label=name)
	# ax.plot(yn,A_or_C, 'o', label=name)

	ax.legend()
	FS=18   #FONT SIZE
	plt.xlabel(out["xlabel"], fontsize=FS)
	plt.ylabel(out["ylabel"], fontsize=FS)
	plt.show()

out["x"]=x.tolist()
out["y"]=yn.tolist()

write_json(out,name+'.json')



		