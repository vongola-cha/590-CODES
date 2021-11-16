
import  torch 

m = torch.nn.Tanh()
input = torch.randn(2)
print(input)
output = m(input)
print(output,m)