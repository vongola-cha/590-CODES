

# #------------------------
# #LOW RES GRAYSCALE IMAGE 
# #------------------------
import numpy as np
import matplotlib.pyplot as plt


#QUICK INFO ON IMAGE
def get_info(image):
    print("------------------------")
    print("IMAGE INFO")
    print("------------------------")
    print("TYPE:",type(image))
    print("SHAPE:",image.shape)
    print("NUMBER OF PIXELS:",image.shape[0]*image.shape[1])
    print("NUMBER OF ENTRIES:",image.size)
    # print("N CHANNELS",image.shape[2])
    print("MIN:", image.min())
    print("MAX:", image.max())
    print("TYPE:",image.dtype)
    print("pixel-1 :", image[0,0])
    # print("image[0:3].shape:", image[0:3].shape)

#SURFACE PLOT
def surface_plot(image):
    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d') #viridis
    ax.plot_surface(xx, yy, image[:,:] ,rstride=1, cstride=1, cmap=plt.cm.gray,linewidth=0)
    plt.show()



# #------------------------
# #LOW RES GRAYSCALE NOISE EXAMPLE
# #------------------------

# # #SINGLE IMAGE
# # print('---------SINGLE------------')
# # #HEIGHT,WIDTH

# # x =(np.random.uniform(0,255,100).reshape(10,10)).astype(int)
# # plt.imshow(x, cmap=plt.cm.gray); plt.show()
# # print(x)
# # print(x.shape)
# # print(x[0,0],x[0,9],x[9,0])
# # print(x[:,2])
# # print(x[2,:])

# # plt.imshow([x[2,:]], cmap=plt.cm.gray); plt.show()


# exit()
# # #BATCH OF 3 GRAYSCALE SAMPLES
# # print('---------BATCH------------')
# # #SAMPLE,HEIGHT,WIDTH

# # X =(np.random.uniform(0,255,4*100).reshape(3,10,10)).astype(int)
# # for i in range(0,X.shape[0]):
# #     plt.imshow(X[i], cmap=plt.cm.gray); plt.show()

# # print(X.shape)



# #------------------------
# #EXPLORE COLORS 
# #------------------------

# #RANDOM
# # x =(np.random.uniform(0,255,3*1).reshape(1,1,3)).astype(int)
# # plt.imshow(x); plt.show()

# #MANUAL
# x=[[[0,0,0]]] #BLACK
# x=[[[255,255,255]]] #WHITE
# x=[[[191, 44, 44]]]  

# plt.imshow(x); plt.show()

# exit()



# #------------------------
# #LOW RES RGB NOISE EXAMPLE
# #------------------------

# #SINGLE IMAGE
# print('---------SINGLE IMAGE------------')

# #HEIGHT,WIDTH,CHANNELS
# x =(np.random.uniform(0,255,3*100).reshape(10,10,3)).astype(int)
# plt.imshow(x); plt.show()
# # print(x)
# print(x.shape)
# # print(x[0,0],x[0,9],x[9,0])
# # print(x[:,2])
# # print(x[2,:])

# # plt.imshow([x[2,:]], cmap=plt.cm.gray); plt.show()

