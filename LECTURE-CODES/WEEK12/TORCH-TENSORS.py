
import  numpy	as	np
import 	torch 


#-----------------------------
#PYTORCH BASIC OPERATIONS 
#-----------------------------

print("CUDA AVAILABLE:",torch.cuda.is_available())

print("-------------------")
print("NUMPY EXAMPLE")
print("-------------------")
A=np.array([[11., 12, 13], [21,22,23]]);
print("A:	")
print(A)
print("A.T")
print(A.T)
print("A.shape  ",A.shape)
print("A[0,0]		",A[0,0])
print("A[:,0]		",A[:,0])
print("A[0,:]		",A[0,:])
print("A[0,0].item()	",A[0,0].item())
print("np.sin(A):		")
print(np.sin(A))


print("-------------------")
print("TORCH TENSOR EXAMPLE")
print("-------------------")
A=torch.tensor(A)	#CONVERT TO TENSOR 
print("A:	")
print(A)
print("A.shape  ",A.shape)
print("A[0,0]		"); print(A[0,0])
print("A[:,0]		"); print(A[:,0])
print("A[0,:]		"); print(A[0,:])
print("A[:,-1]		"); print(A[:,-1])
print("A.T")
print(A.T)
print("A[0,0].item()	",A[0,0].item())
print("torch.sin(A):		")
print("A device",A.device)
print("A dtype",A.dtype)
print("A is_contiguous",A.is_contiguous())
AT=A.T
print("A.T is_contiguous",AT.is_contiguous())
AT=AT.contiguous()
print("A.T is_contiguous",AT.is_contiguous())
print("A is cuda",A.is_cuda)
print("A.tolist	",A.tolist())
print("A.flatten()"); print(A.flatten())
print(torch.sin(A))
print(A.sin())


print("-------------------")
print("TENSORS VIEWS")
print("-------------------")
x=torch.arange(8)
print(x)
x1=x.view(4, 2)
print(x1,x1.view(4, 2).is_contiguous())
x1[0]=1  #CHANGES ORIGINAL
print(x.view(2, 2, 2),x.view(2,2, 2))
print(x.view(2, 2, 2).shape)

x = torch.tensor([[1], [2], [3]])
print(x,x.size())
print(x.expand(3, 4))
print(x.expand(3, 2))
# -1 means not changing the size of that dimension
print(x.expand(-1, 4))   

x = torch.randn(3, 3)
print(x)
print(torch.diagonal(x,-2))
print(torch.diagonal(x,-1))
print(torch.diagonal(x, 0))
print(torch.diagonal(x, 1))
print(torch.diagonal(x, 2))

# print(x.split(1, 0))
print(x.split(1, 1))


print("-------------------")
print("OPERATIONS")
print("-------------------")
A=torch.tensor([[11., 12, 13], [21,22,23]]);
B=torch.tensor([[11., 12, 13]]);
print(A.shape,B.shape)
print(A+1.5)
print("BROADCASTING:",A+B)
print(torch.sigmoid(A))
print(B.sigmoid())
print(A*B)
print(A**2.0)
print(A.tanh())


print("-------------------")
print("COMPUTE GRADIENT")
print("-------------------")
x = torch.tensor(
    [[1.,2.], [3., 4.]],
    requires_grad=True)
y 	= (2*x**3).sum(); y.backward(); 
print(x.grad); print(6*x**2)

x = torch.tensor([[1.,2.], [3., 4.]],
    requires_grad=True)
y = x**3; 	 #FORWARD
y = 2*y; 
y = y.sin() 
y = y.sum() 
y.backward() #BACKWARD
print(x.grad); print(torch.cos(2*x**3)*6*x**2)



