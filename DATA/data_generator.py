
import json
import numpy as np
#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------

#
def write_json(x,name):
	with open(name, "w") as write_file:
		json.dump(x, write_file)

# def write_csv(x,name):

#MISC PARAM
iplot		=	True

name="housing_price";  out={}





#GROUND-TRUTH PARENT FUNCTION
def f(x,name):  #vectorized; array [x1,x2 ... xN] --> f [y1,y2 ... yN]
	if(name=="housing_price"):
		out=500000*np.exp(-(x/3.)**2.0)+250000*np.exp(-((x-10.)/2.5)**2.0)+150000*np.exp(-((x+12)/3)**2.0)
	return out



if(name=="housing_price"):  
	out["xlabel"]="distance_miles"; out["ylabel"]="house_price"



#GENERATE DATA
N=250; xmin=-20; xmax=20
x = np.linspace(xmin,xmax,N)
y =f(x,name)  #PRISTINE DATA
noise=0.075*(max(y)-min(y))*np.random.uniform(-1,1,size=len(x))
yn = y + noise



if(iplot):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.plot(x, yn, 'o', label=name)
	ax.legend()
	FS=18   #FONT SIZE
	plt.xlabel(out["xlabel"], fontsize=FS)
	plt.ylabel(out["ylabel"], fontsize=FS)
	plt.show()

out["x"]=x.tolist()
out["y"]=y.tolist()

write_json(out,name+'.json')

		