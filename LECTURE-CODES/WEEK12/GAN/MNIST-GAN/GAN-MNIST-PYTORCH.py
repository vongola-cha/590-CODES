

#https://medium.com/intel-student-ambassadors/mnist-gan-detailed-step-by-step-explanation-implementation-in-code-ecc93b22dc60


from torchvision import datasets
import torchvision.transforms as transforms
# 1
num_workers = 0
# 2
batch_size = 64
# 3
transform = transforms.ToTensor()
# 4
train_data = datasets.MNIST(root=’’, train=True,
 download=True, transform=transform)
# 5
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
 num_workers=num_workers)


#1 
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()
#2
img = np.squeeze(images[0])
fig = plt.figure(figsize = (3,3)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap=’gray’)