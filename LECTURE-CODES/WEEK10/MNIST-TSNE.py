import numpy as np
import time
import pandas as pd
import seaborn as sns

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


####GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) #NORMALIZE
X=X.reshape(60000,28*28); 

#DOWNSAMPLE 
NKEEP=5000; X=X[0:NKEEP]; Y=Y[0:NKEEP]

#COMPUTE
time_start = time.time()
n_components=3
tsne = TSNE(n_components, verbose=1, perplexity=40, n_iter=300)
X1 = tsne.fit_transform(X)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
print(X1)


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
