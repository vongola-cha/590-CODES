


from SETUP import * 

from keras.models import load_model
from keras.utils.np_utils import to_categorical

model = load_model('model-DFF.h5',compile=True)


# #MAKE PREDICTIONS
# if(batch_size==1): #STOCASTIC
#     t1=[]; yv1=[]; yv2=[]
#     for i in range(B3,B4):
#         samples, targets  = next(val_gen)
#         yv2.append(model.predict(samples)[0,0])
#         yv1.append(targets[0])
#         t1.append(t[max(indices_global[0])+delay+1])
#         # print(i,max(indices_global[0])+delay+1,model.predict(samples)[0,0],targets[0])
if(batch_size!=1): #BATCH
    t1=[]; yv1=[]; yv2=[]
    samples, targets  = next(val_gen)
    #print(samples.shape,targets.shape,len(indices_global),model.predict(samples).shape)
    for i in range(0,len(indices_global)):
        yv2.append(model.predict(samples)[i,0])
        yv1.append(targets[i])
        t1.append(t[max(indices_global[i])+delay+1][0])
        #print(t1[i],yv1[i],yv2[i])

    samples, targets  = next(train_gen)
    # print(samples.shape,targets.shape,len(indices_global),model.predict(samples).shape)
    for i in range(0,len(indices_global)):
        yv2.append(model.predict(samples)[i,0])
        yv1.append(targets[i])
        t1.append(t[max(indices_global[i])+delay+1][0])
        #print(t1[i],yv1[i],yv2[i])




# file_name=



# model = load_model('model-in.h5',compile=True)

# #GET DATA  
# from keras.datasets import mnist
# (x, y), (x_test, y_test) = mnist.load_data()

# #RESHAPE
# y=to_categorical(y); y_test=to_categorical(y_test)
# def reshape_2(x): return x.reshape((x.shape[0],x.shape[1] * x.shape[2])) 
# x = reshape_2(x); x_test = reshape_2(x_test) 
# # print(x.shape,y.shape,x_test.shape,y_test.shape)

# #EVALUATE LOADED MODEL
# print(model.evaluate(x,y,batch_size=len(x)))
# print(model.evaluate(x_test,y_test,batch_size=len(x_test)))

# #SAVE MODEL
# model.save('model-out.h5')   


# # loss = history.history['loss']
# # val_loss = history.history['val_loss']
# # epochs = range(1, len(loss) + 1)
# # plt.figure()
# # plt.plot(epochs, loss, 'bo', label='Training loss')
# # plt.plot(epochs, val_loss, 'b', label='Validation loss')
# # plt.title('Training and validation loss')
# # plt.legend()
# # plt.show()