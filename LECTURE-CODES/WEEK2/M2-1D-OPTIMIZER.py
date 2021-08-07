


#--------------------------------------------------------
#DESCRIPTION
#--------------------------------------------------------
#CODE DOES THE FOLLOWING:

# 1) GENERATES TOY DATA TO FIND THE ROOTS OF (i.e solve f(x)=0)
# 2) IMPLEMENTS 1D SECANT METHOD TO FIND ROOTS 
# 3) VISUALIZES THE PROCESS 

import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt

import 	scipy.optimize 
import 	torch 


#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------

#MISC PARAMETERS
iplot		=	True

#DATA PARAMETERS
a			=	1.0; 
b			=	1.0; 
c			=	1.0; 
d			=	1.0
max_noise	=	0.0

#MESH PARAM
xo			=	c-4. #0.75; 
xf			=	c+4. #0.75;  
N			=	100;
XMID		=	(xf+xo)/2.0
DX			=	(xf-xo)

def f(x): #parent function
	out=x**2
	return out
		
def fp(x, A, xo, w, S): #parameterized function
    return A*f((x-xo)/w)+S

#DEFINE VARIOUS ARRAYS
X1 		= 	np.linspace(xo,xf,N).reshape(N,1)
X2 		= 	np.linspace(XMID-5*DX,XMID+5*DX,10*N).reshape(10*N,1)
YGT1 	= 	fp(X1, a, b, c, d)	#GROUND TRUTH
YGT2 	= 	fp(X2, a, b, c, d)	#GROUND TRUTH
NOISE	=	np.random.uniform(-max_noise,max_noise,N).reshape(N,1)
YN 		= 	YGT1+NOISE		#NOISY DATA





##K=1 CROSS VALIDATION
#indices = np.random.permutation(X1.shape[0])
#CUT=int(0.8*X1.shape[0])
#training_idx, test_idx = indices[:CUT], indices[CUT:]
#XTRAIN, XTEST = X1[training_idx,:], X1[test_idx,:]
#YTRAIN, YTEST = YN[training_idx,:], YN[test_idx,:]

##FITTING MODEL	
#TYPE="LINEAR"
#TYPE="POLYNOMIAL"
#TYPE="LOGISTIC"
#TYPE='ANN'; #layers=[


TYPE="LINEAR" 
#R1 --> R1
def MODEL(x,PARAM):


	if(TYPE=="LINEAR"):
		
		#ERROR CHECK
		if(len(PARAM))>2: print("ERROR: TO MANY PARAM"); exit()
	
		return PARAM[0]+PARAM[1]*x

	if(TYPE=="POLYNOMIAL"):
		out=PARAM[0] #CONSTANT TERM
		for i in range(1,len(PARAM)):
			out=out+PARAM[i]*(x**i)
		
		return out

	if(TYPE=="LOGISTIC"):
		print("LOGISTIC REGRESSION NOT CODED"); exit()

	# if(TYPE=="ANN"):





# def ANN(x,PARAM):
# 	#NOTE THIS DOES 

# 	# #BATCH IMPLEMENTATION
# 	# #print(XTRAIN2.shape,YTRAIN2.shape); # exit()
# 	# def NN_eval(X,submatrices):
# 		out=(X).mm(torch.t(submatrices[0]))+torch.t(submatrices[1]) 
# 		for i in range(2,int(len(submatrices)/2+1)):
# 			j=2*(i-1); #print(j)
# 			out=torch.sigmoid(out)-0.5
# 			out=(out).mm(torch.t(submatrices[j]))+torch.t(submatrices[j+1]) 
# 		return out 


# 	#scalar x 
# 	#array param
# 	#return scalar 
	
# 	#return A*x**2.0+B*x+C



def objective(X,Y,P):
	YPRED=MODEL(X,P)
	SQR_ERROR=(YPRED-Y)**2.0 #COMPUTE ARRAY OF ERRORS AND SQUARE EACH ENTRY 
	MSE=np.mean(SQR_ERROR) 	 #MEAN SQUARE ERROR
	RMSE=MSE**0.5		 #ROOT MEAN SQUARE ERROR 
	return RMSE
	
def optimize_loop(): 

	
	#INITIALIZE FITTING PARAM
	NFIT=2; PLO=-0.1; PHI=0.1;	#NUMBER OF FITTING PARAM
	PARAM=np.random.uniform(PLO,PHI,NFIT) #.reshape(NFIT,1); #print(PARAM.shape)
	
	#ERROR GIVEN CURRENT PARAM
	#e0=objective(XTRAIN,YTRAIN,PARAM)
	#print("RMSE", e0)
	#plt.figure()
	



	plt.ion()  #Enable interactive mode.
	fig,ax = plt.subplots(3,1,figsize=(15,15))
	plt.show()
	label_added =False
	# ax[1] = plt.axes(projection='3d')

	for step in range(0,1000):
		YPRED=MODEL(XTRAIN,PARAM)
		YPRED_TEST=MODEL(XTEST,PARAM)

		RMSE_TRAIN=objective(XTRAIN,YTRAIN,PARAM)
		RMSE_TEST=objective(XTEST,YTEST,PARAM)

		#PLOT DATA SET
		if(step%10==0):

			ax[0].clear()

			ax[0].plot(XTRAIN, YTRAIN, 'bo', label="Training data")
			ax[0].plot(XTEST, YTEST, 'ro',  label="testing data")
			ax[0].plot(XTRAIN,YPRED,'*')
			ax[0].plot(XTEST,YPRED_TEST,'*')

			ax[0].legend() 
			ax[0].set_xlabel('x')
			ax[0].set_ylabel('y')


			#HISTORY
			# ax[1].plot(XTRAIN, YTRAIN, 'bo', label="Training data")
			# ax[1].plot(XTEST, YTEST, 'ro',  label="testing data")
			# ax[1].plot(XTRAIN,YPRED,'g-', linewidth=1, alpha=0.3)
			# # ax[2].plot(XTRAIN,YPRED,'g-', linewidth=2)

			# ax[1].set_xlabel('x')
			# ax[1].set_ylabel('y')


			# ax[1].scatter3D(param[0], param[1], RMSE_TRAIN)



			if(step==25): 
				ax[2].clear(); #label_added=False
				ax[2].plot(step,RMSE_TRAIN,'ro', label="TRAINING RMSE") 
				ax[2].plot(step,RMSE_TEST,'bo', label="TEST RMSE") 
				ax[2].legend() 

			ax[2].plot(step,RMSE_TRAIN,'ro', label="TRAINING RMSE") 
			ax[2].plot(step,RMSE_TEST,'bo', label="TEST RMSE") 
			ax[2].set_xlabel('Optimizer step')
			ax[2].set_ylabel('RMSE (RED=TRAINING) (BLUE=TEST)')
			if not label_added:
				ax[1].legend() 
				ax[2].legend() 
				label_added=True 

			plt.draw()

			plt.pause(0.01)


		#COMPUTE GRADIENT dE/dPARAM_i USING FINITE DIFFERENC 
		dp=0.001; grad=[]
		PARAM_P1=np.copy(PARAM); PARAM_M1=np.copy(PARAM);
		# print(PARAM)
		

		for i in range(0,len(PARAM)):
			#PERTURB
			
	 		#CENTRAL FINITE DIFFERENCE
			#print(PARAM[i])
			PARAM_P1[i]=PARAM[i]+dp; PARAM_M1[i]=PARAM[i]-dp
			#print(i,PARAM_P1[i],PARAM[i],PARAM[i]+dp,PARAM_P1[0])
			#exit()
			grad.append((objective(XTRAIN,YTRAIN,PARAM_P1)-objective(XTRAIN,YTRAIN,PARAM_M1))/dp)
		print(grad)

		LR=0.01
		#MAKE STEP USING GRADIENT DESCENT 
		PARAM=PARAM-LR*np.array(grad)

		
		
	#plt.show()
		
	exit()		
	
	return 





optimize_loop()



exit()




#LOW DIMENSIONAL REGRESSION WITHOUT KERAS,TENSORFLOW, OR PYTORCH

#INCREASE FAMILIARITY WITH JSON 
#INCREASE FAMILIARITY WITH PYTHON OBEJECTS
#GENERAL UNDERSTANDING OF REGRESSION
#INCREASE PYTHON STANDARDS 
#VARIABLE SCOPE 


#HOMEWORK: GENERALIZE TO NDIM
	#CODE ANN MODEL 
	#FIT 2D PLANE AND VISUALIZE
	#FIT 2D SURFACE AND VISUALIZE 
	#FIT 3D  F(x,y,z) --> scalar (use color) 

#CLASS EXAMPLE: MPG
#HOMEWORK EXAMPLE: MAYBE GENERATE TOY MODELS DATA FILES
	#DONT JUST DO MATH 
	#1D+2D+3D	
	#LINEAR 1D
	#LINEAR 2D
	#POLY

#TYPES OF NOISE+UNCERTRAINTY
#CLASSES 
#MODEL COMPLEXITY



