#-----------------------------------
#DICE EXAMPLE
#----------------------------------
from itertools import permutations
from itertools import product
import numpy as np
import matplotlib.pyplot as plt


N_DIE=[]; ENTROPY=[]
for N_DICE in range(1,8):
	print("--------",N_DICE,"--------")
	#CREATE ALL PERMUTATIONS
	myList = list(product(range(1,7), repeat=N_DICE))  

	#-------------------
	#DISTINGUISHABLE DICE: MACROSTATE --> (1,2) != (2,1)
	#-------------------
	Nmicro=len(myList)
	pi=np.ones(Nmicro)/Nmicro
	S=-sum(pi*np.log(pi))
	print("DISTINGUISHABLE",S,np.log(Nmicro),1.792*N_DICE)

	#-------------------
	#INDISTINGUISHABLE: USE SUM AS MACOSTATE  (1,2)=(2,1) --> state-3
	#-------------------

	#FIND NUMBER OF UNIQUIE STATES
	uniq=[]
	for i in range(0,len(myList)):
		S=sum(myList[i])
		if(S not in uniq): uniq.append(S)
	print("N UNIQUE SUMS",len(uniq))

	#COMPUTE PROBABLITIES FOR MAC STATES
	Nmicro=0; states=np.zeros(len(uniq))
	for i in range(0,len(myList)):
		S=sum(myList[i])
		#print(i,myList[i],S)
		Nmicro=Nmicro+1
		states[S-N_DICE]=states[S-N_DICE]+1
	pi=states/states.sum()
	S=-sum(pi*np.log(pi))
	print("STATES:",states)
	print("Pi=",np.round(pi,4) )
	print("ENTROPY",S)
	print("ENTROPY/DICE",S/N_DICE)
	N_DIE.append(N_DICE)
	ENTROPY.append(S)

#PLOT
fig = plt.figure(figsize=(20,12))
ax  = fig.add_subplot(111)
plt.plot(N_DIE,ENTROPY,'bo-')
ax.set_xlabel('number of dice')
ax.set_ylabel('entropy of system')
plt.show()

