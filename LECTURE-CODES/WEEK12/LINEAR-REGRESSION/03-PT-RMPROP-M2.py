
#pip install torchsummary 


import  numpy           as  np
import  matplotlib.pyplot   as  plt
import  scipy.optimize 
import  torch 
from    torchsummary import summary

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
XTRAIN=torch.tensor(XTRAIN).float();  
YTRAIN=torch.tensor(YTRAIN).float(); 
XTEST=torch.tensor(XTEST).float();  
YTEST=torch.tensor(YTEST).float();  
# print(XTRAIN.shape,YTRAIN.shape); exit()

#---------------------------
#MODEL
#---------------------------

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize=1, outputSize=1):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
       # print(out); #exit()
        return out

model = linearRegression(inputSize=1, outputSize=1)

print(model)
print(summary(model,(1,)))
exit()

#---------------------------
#TRAIN WITH PYTORCH
#---------------------------

#INITIALIZE OPTIMIZER

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.01, alpha = 0.9)

#OPTIMIZATION LOOP
itr=1; iter_max=150
plt.figure(); time=[]; err=[]
while(itr<iter_max):  

    #EVALUATE LOSS
    optimizer.zero_grad();  #RESET GRADIENTS 
    ypred=model(XTRAIN)
    loss = criterion(ypred, YTRAIN)
    loss.backward();    #COMPUTE DERIVATIVES
    optimizer.step()

    #COMPUTE TEST RMSE
    test_out=model(XTEST)
    test_RMSE=criterion(test_out, YTEST)
    print(itr,loss.item(),test_RMSE.item(),loss.item()/test_RMSE.item())
    itr=itr+1
    
    #ERROR
    plt.plot(itr,loss.item(),'ro')
    plt.plot(itr,test_RMSE.item(),'bo')
    plt.pause(0.005)
plt.show()  


#PLOT
plt.figure()
plt.plot(XTRAIN, YTRAIN, 'o', label="Training data")
plt.plot(XTEST, YTEST, 'o',  label="testing data")
plt.plot(XTRAIN,model(XTRAIN).detach().numpy() , 'b-', label="model fit")
plt.legend()
plt.show()

