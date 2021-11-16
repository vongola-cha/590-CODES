
import  numpy           as  np
import  matplotlib.pyplot   as  plt
import  scipy.optimize 
import  torch 

#--------------------------- 
#LOAD DATA AND SETUP
#--------------------------- 

#LOAD DATA
DATA=np.load('X-Y.npy')
# plt.figure(); plt.plot(DATA[:,0], DATA[:,1], 'o'); plt.show()
N=DATA.shape[0]
X=DATA[:,0].reshape(N,1); Y=DATA[:,1].reshape(N,1)

#VALIDATION
indices = np.random.permutation(X.shape[0])
CUT=int(0.8*X.shape[0]); print(CUT)
training_idx, test_idx = indices[:CUT], indices[CUT:]
XTRAIN, XTEST = X[training_idx,:], X[test_idx,:]
YTRAIN, YTEST = Y[training_idx,:], Y[test_idx,:]

# PLOT DATA SET
if(False):
    plt.figure()
    plt.plot(XTRAIN, YTRAIN, 'o', label="Training data")
    plt.plot(XTEST, YTEST, 'o',  label="testing data")
    plt.legend()
    plt.show()

#CONVERT DATA TO TOURCH TENSORS
XTRAIN=torch.tensor(XTRAIN);  
YTRAIN=torch.tensor(YTRAIN); 
XTEST=torch.tensor(XTEST);  
YTEST=torch.tensor(YTEST);  

#---------------------------
#MODEL
#---------------------------

#FITTING PARAMETERS
param=torch.tensor([0.,1]) #.type(self.dtype)
param=param.contiguous() #force to be contigous in memory
param.requires_grad = True

#DEFINE MODEL
def model_eval(X,param):
    out=(X)*param[0]+param[1] 
    return out 

#---------------------------
#TRAIN WITH PYTORCH
#---------------------------

#INITIALIZE OPTIMIZER
optimizer = torch.optim.RMSprop([param], lr = 0.01, alpha = 0.9)

#OPTIMIZATION LOOP
itr=1; iter_max=125
plt.figure(); time=[]; err=[]
while(itr<iter_max):  

    #EVALUATE LOSS
    optimizer.zero_grad();  #RESET GRADIENTS 
    loss=0.0 
    out=model_eval(XTRAIN,param)
    out=out.reshape(out.shape[0])
    loss=(((out-YTRAIN[:,0])**2.0).sum()/N)**0.5  #RMSE
    loss.backward();    #COMPUTE DERIVATIVES
    optimizer.step()

    #COMPUTE TEST RMSE
    test_out=model_eval(XTEST,param)
    test_out=test_out.reshape(test_out.shape[0])
    test_RMSE=((((test_out-YTEST[:,0])**2.0).sum()/YTEST.shape[0])**0.5).item()
    
    print(itr,loss.item(),test_RMSE,loss.item()/test_RMSE)
    itr=itr+1
    
    #ERROR
    plt.plot(itr,loss.item(),'ro')
    plt.plot(itr,test_RMSE,'bo')
    plt.pause(0.005)
plt.show()  

print("FINAL PARAM",param)

#PLOT
plt.figure()
plt.plot(XTRAIN, YTRAIN, 'o', label="Training data")
plt.plot(XTEST, YTEST, 'o',  label="testing data")
plt.plot(XTRAIN,model_eval(XTRAIN,param).detach().numpy() , 'b-', label="model fit")
plt.legend()
plt.show()

