

import 	numpy 			as	np
import 	matplotlib.pyplot 	as 	plt

import 	scipy.optimize 
import 	torch 

#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------

#MISC PARAM
iplot		=	True

#DATA PARAM
a		=	1.0; 
b		=	1.0; 
c		=	1.0; 
d		=	1.0
max_noise	=	0.35

#MESH PARAM
xo		=	c-4. #0.75; 
xf		=	c+4. #0.75;  
N		=	100;
XMID		=	(xf+xo)/2.0
DX		=	(xf-xo)


		
def F1(x, A, B, C, D):
#    return (A*np.exp(-B * (x-C)**2.0) + D) #*x**2
    return (A*np.sin(B *(x-C)) + D) #*x**2
    
#DEFINE VARIOUS ARRAYS
X1 		= 	np.linspace(xo,xf,N).reshape(N,1)
X2 		= 	np.linspace(XMID-5*DX,XMID+5*DX,10*N).reshape(10*N,1)
YGT1 		= 	F1(X1, a, b, c, d)	#GROUND TRUTH
YGT2 		= 	F1(X2, a, b, c, d)	#GROUND TRUTH
NOISE		=	np.random.uniform(-max_noise,max_noise,N).reshape(N,1)
YN 		= 	YGT1+NOISE		#NOISY DATA

#K=1 CROSS VALIDATION
indices = np.random.permutation(X1.shape[0])
CUT=int(0.8*X1.shape[0])
training_idx, test_idx = indices[:CUT], indices[CUT:]
XTRAIN, XTEST = X1[training_idx,:], X1[test_idx,:]
YTRAIN, YTEST = YN[training_idx,:], YN[test_idx,:]

#PLOT DATA SET
if(iplot):
	plt.figure()
	plt.plot(XTRAIN, YTRAIN, 'o', label="Training data")
	plt.plot(XTEST, YTEST, 'o',  label="testing data")
	plt.plot(X2, YGT2, 'r-', label="Ground Truth")
	plt.legend()
	plt.show()

#exit()
#--------------------------------------------------------
#QUADRATIC REGRESSION
#--------------------------------------------------------

#FITTING MODEL
def F2(x, A, B, C):
    return A*x**2.0+B*x+C

#USE scipy.optimize TO NUMERICALLY FIT A,B,C
popt, pcov = scipy.optimize.curve_fit(F2, XTRAIN[:,0], YTRAIN[:,0])
print("QUAD PARAM	:	",popt)

#--------------------------------------------------------
#NEURAL NETWORK REGRESSION (PYTORCH)
#--------------------------------------------------------

#ANN PARAM
layers		=	[1,10,10,1] #[1,6,6,6,1]
max_rand_wb	=	1.0 
activation	=	"SIGMOID"
LR		=	0.01
GAMMA_L1	=	0.0		#LASO 
GAMMA_L2	=	0.001		#RIDGE
iter_max	=	100		#MAX NUMBER OF TRAINING ITERATIONS

#CALCULATE NUMBER OF FITTING PARAMETERS FOR SPECIFIED NN 
nfit=0; 
for i in range(1,len(layers)):  nfit=nfit+layers[i-1]*layers[i]+layers[i]
		
#RANDOMIZE NN FITTING PARAM (WEIGHTS AND BIAS)
WB		=	np.random.uniform(-max_rand_wb,max_rand_wb,nfit)

#WRITE TO SCREEN
print("NFIT		:	",nfit)
print("WB		:	",WB)

#TAKES A LONG VECTOR W OF WEIGHTS AND BIAS AND RETURNS WEIGHT AND BIAS SUBMATRICES
def extract_submatrices(WB):
	submatrices=[]; K=0
	for i in range(0,len(layers)-1):
		#FORM RELEVANT SUB MATRIX FOR LAYER-N
		Nrow=layers[i+1]; Ncol=layers[i] #+1
		w=np.array(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
		K=K+Nrow*Ncol; #print i,k0,K
		Nrow=layers[i+1]; Ncol=1; #+1
		b=np.transpose(np.array([WB[K:K+Nrow*Ncol]])) #unpack/ W 
		K=K+Nrow*Ncol; #print i,k0,K
		submatrices.append(w); submatrices.append(b)

	#CONVERT TO TORCH TENSORS AND SET GRAD
	for i in range(0,len(submatrices)):
			submatrices[i]=torch.tensor(submatrices[i]) #.type(self.dtype)
			submatrices[i]=submatrices[i].contiguous() #force to be contigous in memory
			submatrices[i].requires_grad = True
#			print(submatrices[i].is_contiguous())
			print("MATRIX SHAPES	:	"+str(submatrices[i].shape))

	return submatrices
		
submatrices=extract_submatrices(WB)


#CONVERT DATA TO TOURCH TENSORS
XTRAIN2=torch.tensor(XTRAIN);  
YTRAIN2=torch.tensor(YTRAIN); 
XTEST2=torch.tensor(XTEST);  
YTEST2=torch.tensor(YTEST);  
X22=torch.tensor(X2);  

#exit()

#BATCH IMPLEMENTATION
#print(XTRAIN2.shape,YTRAIN2.shape); # exit()
def NN_eval(X,submatrices):
	out=(X).mm(torch.t(submatrices[0]))+torch.t(submatrices[1]) 
	for i in range(2,int(len(submatrices)/2+1)):
		j=2*(i-1); #print(j)
		out=torch.sigmoid(out)-0.5
		out=(out).mm(torch.t(submatrices[j]))+torch.t(submatrices[j+1]) 
	return out 

#SET UP PYTORCH OPTIMZER FOR LBFGS
optimizer=torch.optim.LBFGS(submatrices, max_iter=20, lr=LR) 	


#OBJECTIVE FUNCTION
def closure():
	global loss,XTRAIN2,YTRAIN2,submatrices  
	
	optimizer.zero_grad(); 	#RESET GRADIENTS 
	
	loss=0.0 
	out=NN_eval(XTRAIN2,submatrices)
	out=out.reshape(out.shape[0])

	loss=(((out-YTRAIN2[:,0])**2.0).sum()/N)**0.5  #RMSE
#	print(out.shape,YTRAIN2[:,0].shape,loss); exit()
	#LASSO 
	L1=0
	for i in range(0,len(submatrices)):
		L1=L1+((submatrices[i]**2.0)**0.5).sum()
	loss=loss+GAMMA_L1*L1
	
	#RIDGE
	L2=0
	for i in range(0,len(submatrices)):
		L2=L2+(submatrices[i]**2.0).sum()
	loss=loss+GAMMA_L2*L2

	loss.backward();	#COMPUTE DERIVATIVES
	return loss


#OPTIMIZATION LOOP
itr=1; #iter_max=100
plt.figure(); time=[]; err=[]
while(itr<iter_max):  

	optimizer.step(closure)
	
	#COMPUTE TEST RMSE
	test_out=NN_eval(XTEST2,submatrices)
	test_out=test_out.reshape(test_out.shape[0])
	test_RMSE=((((test_out-YTEST2[:,0])**2.0).sum()/YTEST2.shape[0])**0.5).item()
	
	print(itr,loss.item(),test_RMSE,loss.item()/test_RMSE)
	itr=itr+1

	#WEIGHTS AND BIAS
	WB_TMP=[]
	for l in range(0,len(submatrices)):
		for j in range(0,len(submatrices[l][0])): #len(w1[0])=number of columns
			for i in range(0,len(submatrices[l])): #build down the each row then accros
				WB_TMP.append(submatrices[l][i][j].item())
	plt.plot(WB_TMP,'bo')
	plt.pause(0.05)
	plt.clf()
	
	#ERROR
	# plt.plot(itr,loss.item(),'ro')
	# plt.plot(itr,test_RMSE,'bo')
	# plt.pause(0.005)

plt.show()	


if(iplot):
	plt.figure()
	plt.plot(XTRAIN, YTRAIN, 'o', label="Training data")
	plt.plot(XTEST, YTEST, 'o',  label="testing data")
	plt.plot(X2, YGT2, 'r-', label="Ground Truth")
	plt.plot(X2, F2(X2,*popt), 'k-', label="Quadratic fit")
	plt.plot(X22,NN_eval(X22,submatrices).detach().numpy() , 'b-', label="ANN fit")
	plt.legend()
	plt.show()


exit()

