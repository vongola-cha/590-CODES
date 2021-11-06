import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

####GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) #NORMALIZE
X=X.reshape(60000,28*28); #print(X[0])

#COMPUTE PCA
from sklearn.decomposition import PCA
n_components=5
pca = PCA(n_components=n_components)
# pca.fit(X)
X1=pca.fit_transform(X)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

#PAIR PLOT
if(n_components<=5):
	df = pd.DataFrame(X1)
	df['Y']=Y

	sns.pairplot(
	df, 
	diag_kind='kde', 
	kind="hist", 
	palette=sns.color_palette("hls", 10),
	hue='Y')   
	plt.show()
	exit()




# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x=0, y=1,
#     hue=Y,
#     palette=sns.color_palette("hls", 10),
#     data=df,
#     legend="full",
#     alpha=0.3
# )
# plt.show()


# print(df.head())
# exit()

# print('PCA')
# print(pca.components_)
# print(X1.shape)

# df['pca-one'] = X1[:,0]
# df['pca-two'] = X1[:,1] 
# df['pca-three'] = X1[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# exit()


# plt.plot(X1[:,0], X1[:,1],'o', label='target')

# # plt.legend()
# # plt.xlabel('Observation number after given time steps')
# # plt.ylabel('Sunspots scaled')
# # plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
# plt.show()



# # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# print(X.shape)

# exit()

# #QUICK INFO ON IMAGE
# def get_info(image):
# 	print("\n------------------------")
# 	print("INFO")
# 	print("------------------------")
# 	print("SHAPE:",image.shape)
# 	print("MIN:",image.min())
# 	print("MAX:",image.max())
# 	print("TYPE:",type(image))
# 	print("DTYPE:",image.dtype)
# #	print(DataFrame(image))

# get_info(X); #get_info(train_labels)
