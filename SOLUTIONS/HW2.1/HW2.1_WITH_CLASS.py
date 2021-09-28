#--------------------------------
# UNIVARIABLE REGRESSION EXAMPLE
#    -USING SciPy FOR OPTIMIZATION
#--------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
# from   scipy.optimize import minimize

# #------------------------
# #DOUBLE CHECK PART-1 OF HW2.1
# #------------------------

# x=np.array([[3],[1],[4]])
# y=np.array([[2,5,1]])

# A=np.array([[4,5,2],[3,1,5],[6,4,3]])
# B=np.array([[3,5],[5,2],[1,4]])
# print(x.shape,y.shape,A.shape,B.shape)
# print(np.matmul(x.T,x))
# print(np.matmul(y,x))
# print(np.matmul(x,y))
# print(np.matmul(A,x))
# print(np.matmul(A,B))
# print(B.reshape(6,1))
# print(B.reshape(1,6))

#------------------------
#CODE PARAMETERS
#------------------------

#USER PARAMETERS
IPLOT=True
INPUT_FILE='weight.json'
FILE_TYPE="json"

PARADIGM='batch'
# PARADIGM='mini_batch'
# PARADIGM='stocastic'

model_type="logistic"; NFIT=4; X_KEYS=['x']; Y_KEYS=['y']

#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]

#------------------------
#OPTIMIZER FUNCTION
#------------------------
def minimizer(f,x0, algo='MOM', LR=0.01):
	global epoch,epochs, loss_train,loss_val 
	# x0=initial guess, (required to set NDIM)
	# algo=GD or MOM
	# LR=learning rate for gradient decent

	#PARAM
	t=1					#ITERATION COUNTER
	dx=0.0001			#STEP SIZE FOR FINITE DIFFERENCE
	tmax=5000			#MAX NUMBER OF ITERATION
	tol=10**-10			#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	ICLIP=False

	NDIM=len(x0)		#DIMENSION OF OPTIIZATION PROBLEM

	xi=x0				#INITIAL GUESS
	dx_m1=0 			#INITIALZE FOR MOMENTUM ALGORITHM
	alpha=0.5  			#EXPONENTIAL DECAY FACTOR FOR MOMENTUM ALGO

	if(PARADIGM=='stocastic'):
		LR=0.002; tmax=30000; ICLIP=True

	#OPTIMIZATION LOOP
	while(t<=tmax):

		#-------------------------
		#DATASET PARITION BASED ON TRAINING PARADIGM
		#-------------------------
		if(PARADIGM=='batch'):
			if(t==1): index_2_use=D.train_idx
			if(t>1):  epoch+=1

		if(PARADIGM=='mini_batch'):
			#50-50 batch size hard coded
			if(t==1): 
				#DEFINE BATCHS
				batch_size=int(D.train_idx.shape[0]/2)
				#BATCH-1
				index1 = np.random.choice(D.train_idx, batch_size, replace=False)  
				index_2_use=index1; #epoch+=1
				#BATCH-2
				index2 = []
				for i1 in D.train_idx:
					if(i1 not in index1): index2.append(i1)
				index2=np.array(index2)
			else: 
				#SWITCH EVERY OTHER ITERATION
				if(t%2==0):
					index_2_use=index1
				else:
					index_2_use=index2
					epoch+=1

		if(PARADIGM=='stocastic'):
			if(t==1): counter=0;
			if(counter==D.train_idx.shape[0]): 
				counter=0;  epoch+=1 #RESET 
			else: 
				counter+=1
			index_2_use=counter

		#-------------------------
		#NUMERICALLY COMPUTE GRADIENT 
		#-------------------------
		df_dx=np.zeros(NDIM);	#INITIALIZE GRADIENT VECTOR
		for i in range(0,NDIM):	#LOOP OVER DIMENSIONS

			dX=np.zeros(NDIM);  #INITIALIZE STEP ARRAY
			dX[i]=dx; 			#TAKE SET ALONG ith DIMENSION
			xm1=xi-dX; 			#STEP BACK
			xp1=xi+dX; 			#STEP FORWARD 

			#CENTRAL FINITE DIFF
			grad_i=(f(xp1,index_2_use)-f(xm1,index_2_use))/dx/2

			# CLIP GRADIENTS IF NEEDED
			if(ICLIP):
				max_grad=10
				if(grad_i>max_grad):  grad_i=max_grad
				if(grad_i<-max_grad): grad_i=-max_grad

			# UPDATE GRADIENT VECTOR 
			df_dx[i]=grad_i 
			
		#TAKE A OPTIMIZER STEP
		if(algo=="GD"):  xip1=xi-LR*df_dx 
		if(algo=="MOM"): 
			step=LR*df_dx+alpha*dx_m1
			xip1=xi-step
			dx_m1=step

		# #EARLY STOPPING
		# if(t==3000): break

		#REPORT AND SAVE DATA FOR PLOTTING
		if(t%1==0):
			D.predict(xi)	#MAKE PREDICTION FOR CURRENT PARAMETERIZATION
			# print(t,"	",xi,"	",D.MSE_T,"	",D.MSE_V) 
			print(t,"	",epoch,"	",D.MSE_T,"	",D.MSE_V) 

			#UPDATE
			epochs.append(epoch); 
			loss_train.append(D.MSE_T);  loss_val.append(D.MSE_V);

			#STOPPING CRITERION (df=change in objective function)
			df=np.absolute(f(xip1,index_2_use)-f(xi,index_2_use))
			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break

		xi=xip1 #UPDATE
		t=t+1

	return xi

#------------------------
#DATA CLASS
#------------------------

class DataClass:

    #INITIALIZE
	def __init__(self,FILE_NAME):

		if(FILE_TYPE=="json"):

			#READ FILE
			with open(FILE_NAME) as f:
				self.input = json.load(f)  #read into dictionary

			#CONVERT DICTIONARY INPUT AND OUTPUT MATRICES #SIMILAR TO PANDAS DF   
			X=[]; Y=[]
			for key in self.input.keys():
				if(key in X_KEYS): X.append(self.input[key])
				if(key in Y_KEYS): Y.append(self.input[key])

			#MAKE ROWS=SAMPLE DIMENSION (TRANSPOSE)
			self.X=np.transpose(np.array(X))
			self.Y=np.transpose(np.array(Y))

			#EXTRACT AGE<18
			if(model_type=="linear"):
				self.Y=self.Y[self.X[:]<18]; 
				self.X=self.X[self.X[:]<18]; 

			#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
			self.XMEAN=np.mean(self.X,axis=0); self.XSTD=np.std(self.X,axis=0) 
			self.YMEAN=np.mean(self.Y,axis=0); self.YSTD=np.std(self.Y,axis=0) 
		else:
			raise ValueError("REQUESTED FILE-FORMAT NOT CODED"); 

	def report(self):
		print("--------DATA REPORT--------")
		print("X shape:",self.X.shape)
		print("X means:",self.XMEAN)
		print("X stds:" ,self.XSTD)
		print("Y shape:",self.Y.shape)
		print("Y means:",self.YMEAN)
		print("Y stds:" ,self.YSTD)

	def partition(self,f_train=0.8, f_val=0.15,f_test=0.05):
		#TRAINING: 	 DATA THE OPTIMIZER "SEES"
		#VALIDATION: NOT TRAINED ON BUT MONITORED DURING TRAINING
		#TEST:		 NOT MONITORED DURING TRAINING (ONLY USED AT VERY END)

		if(f_train+f_val+f_test != 1.0):
			raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

		#PARTITION DATA
		rand_indices = np.random.permutation(self.X.shape[0])
		CUT1=int(f_train*self.X.shape[0]); 
		CUT2=int((f_train+f_val)*self.X.shape[0]); 
		self.train_idx, self.val_idx, self.test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]

	def normalize(self):
		self.X=(self.X-self.XMEAN)/self.XSTD 
		self.Y=(self.Y-self.YMEAN)/self.YSTD  

	def model(self,x,p):
		if(model_type=="linear"):   return  p[0]*x+p[1]  
		if(model_type=="logistic"): return  p[0]+p[1]*(1.0/(1.0+np.exp(-(x-p[2])/(p[3]+0.0001))))

	def predict(self,p):
		self.YPRED_T=self.model(self.X[self.train_idx],p)
		self.YPRED_V=self.model(self.X[self.val_idx],p)
		self.YPRED_TEST=self.model(self.X[self.test_idx],p)
		self.MSE_T=np.mean((self.YPRED_T-self.Y[self.train_idx])**2.0)
		self.MSE_V=np.mean((self.YPRED_V-self.Y[self.val_idx])**2.0)

	def un_normalize(self):
		self.X=self.XSTD*self.X+self.XMEAN 
		self.Y=self.YSTD*self.Y+self.YMEAN 
		self.YPRED_T=self.YSTD*self.YPRED_T+self.YMEAN 
		self.YPRED_V=self.YSTD*self.YPRED_V+self.YMEAN 
		self.YPRED_TEST=self.YSTD*self.YPRED_TEST+self.YMEAN 

	#------------------------
	#DEFINE LOSS FUNCTION
	#------------------------
	def loss(self,p,index_2_use):
		errors=self.model(self.X[index_2_use],p)-self.Y[index_2_use]  #ERROR VECTOR
		training_loss=np.mean(errors**2.0)		#MSE
		return training_loss

	def fit(self):
		global p_final

		#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
		po=np.random.uniform(2,1.,size=NFIT)

		#TRAIN MODEL USING SCIPY MINIMIZ 
		p_final=minimizer(self.loss,po)		
		print("OPTIMAL PARAM:",p_final)
		self.predict(p_final)

		#PLOT TRAINING AND VALIDATION LOSS AT END
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(epochs, loss_train, 'o', label='Training loss')
			ax.plot(epochs, loss_val, 'o', label='Validation loss')
			plt.xlabel('epochs', fontsize=18)
			plt.ylabel('loss', fontsize=18)
			plt.legend()
			plt.show()

	#FUNCTION PLOTS
	def plot_1(self,xla='x',yla='y'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.X[self.train_idx]    , self.Y[self.train_idx],'o', label='Training') 
			ax.plot(self.X[self.val_idx]      , self.Y[self.val_idx],'x', label='Validation') 
			ax.plot(self.X[self.test_idx]     , self.Y[self.test_idx],'*', label='Test') 
			ax.plot(self.X[self.train_idx]    , self.YPRED_T,'.', label='Model') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()

	#PARITY PLOT
	def plot_2(self,xla='y_data',yla='y_predict'):
		if(IPLOT):
			fig, ax = plt.subplots()
			ax.plot(self.Y[self.train_idx]  , self.YPRED_T,'*', label='Training') 
			ax.plot(self.Y[self.val_idx]    , self.YPRED_V,'*', label='Validation') 
			ax.plot(self.Y[self.test_idx]    , self.YPRED_TEST,'*', label='Test') 
			plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
			plt.show()

#------------------------
#MAIN (RUNNER)
#------------------------
D=DataClass(INPUT_FILE)				#INITIALIZE DATA OBJECT 
D.report()							#BASIC DATA PRESCREENING

D.partition()						#SPLIT DATA
D.normalize()						#NORMALIZE
D.fit()
D.plot_1()							#PLOT DATA
D.plot_2()							#PLOT DATA

D.un_normalize()					#UN NORMALIZE
D.plot_1(xla='age',yla='weight')	#PLOT DATA
D.plot_2()							#PLOT DATA
