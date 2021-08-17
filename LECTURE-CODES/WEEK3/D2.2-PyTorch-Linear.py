

import  numpy           as  np
import  matplotlib.pyplot   as  plt

import  scipy.optimize 
import  torch 

#--------------------------------------------------------
##GENERATE DATA
#--------------------------------------------------------

#MISC PARAM
iplot       =   True

#DATA PARAM
a       =   1.0; 
b       =   1.0; 
max_noise   =   0.35


#GROUND TRUTH       
def F1(x, A, B):
    return A*x+B
    

#MESH PARAM
xo      =   -4. #0.75; 
xf      =   +4. #0.75;  
N       =   100;
XMID    =   (xf+xo)/2.0
DX      =   (xf-xo)

#DEFINE VARIOUS ARRAYS
X1      =   np.linspace(xo,xf,N).reshape(N,1)
X2      =   np.linspace(XMID-1*DX,XMID+1*DX,2*N).reshape(2*N,1)
YGT1    =   F1(X1, a, b)  #GROUND TRUTH
YGT2    =   F1(X2, a, b)  #GROUND TRUTH
NOISE   =   np.random.uniform(-max_noise,max_noise,N).reshape(N,1)
YN      =   YGT1+NOISE      #NOISY DATA

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

#--------------------------------------------------------
#NEURAL NETWORK REGRESSION (PYTORCH)
#--------------------------------------------------------

#CONVERT DATA TO TOURCH TENSORS
XTRAIN2=torch.tensor(XTRAIN);  
YTRAIN2=torch.tensor(YTRAIN); 
XTEST2=torch.tensor(XTEST);  
YTEST2=torch.tensor(YTEST);  
X22=torch.tensor(X2);  

#FITTING PARAMETERS
param=torch.tensor([0.,1]) #.type(self.dtype)
param=param.contiguous() #force to be contigous in memory
param.requires_grad = True
#print(param); exit()


#BATCH IMPLEMENTATION
#print(XTRAIN2.shape,YTRAIN2.shape); # exit()
def model_eval(X,param):
    out=(X)*param[0]+param[1] 
    return out 

#SET UP PYTORCH OPTIMZER FOR LBFGS
optimizer=torch.optim.LBFGS([param], max_iter=20, lr=0.001)    

#OBJECTIVE FUNCTION
def closure():
    global loss,XTRAIN2,YTRAIN2,param  
    
    optimizer.zero_grad();  #RESET GRADIENTS 
    
    loss=0.0 
    out=model_eval(XTRAIN2,param)
    out=out.reshape(out.shape[0])

    loss=(((out-YTRAIN2[:,0])**2.0).sum()/N)**0.5  #RMSE

    loss.backward();    #COMPUTE DERIVATIVES
    return loss


#OPTIMIZATION LOOP
itr=1; iter_max=100
plt.figure(); time=[]; err=[]
while(itr<iter_max):  

    optimizer.step(closure)
    
    #COMPUTE TEST RMSE
    test_out=model_eval(XTEST2,param)
    test_out=test_out.reshape(test_out.shape[0])
    test_RMSE=((((test_out-YTEST2[:,0])**2.0).sum()/YTEST2.shape[0])**0.5).item()
    
    print(itr,loss.item(),test_RMSE,loss.item()/test_RMSE)
    itr=itr+1
    
    #ERROR
    plt.plot(itr,loss.item(),'ro')
    plt.plot(itr,test_RMSE,'bo')
    plt.pause(0.005)

plt.show()  


if(iplot):
    plt.figure()
    plt.plot(XTRAIN, YTRAIN, 'o', label="Training data")
    plt.plot(XTEST, YTEST, 'o',  label="testing data")
    plt.plot(X2, YGT2, 'r-', label="Ground Truth")
    plt.plot(X22,model_eval(X22,param).detach().numpy() , 'b-', label="ANN fit")
    plt.legend()
    plt.show()


exit()
