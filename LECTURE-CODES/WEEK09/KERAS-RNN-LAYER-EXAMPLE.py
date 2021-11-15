#EXAMPLE MODIFIED FROM: 
#https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
import matplotlib.pyplot as plt

#USER PARAM
USE_DENSE=True
RETURN_SEQUENCES=True
N_OUTPUT=2
USE_INPUT=2

#CHOOSE WHICH INPUT TO USE
if(USE_INPUT==1):
    x = np.array([1,2,3]); x = np.reshape(x,(1, 3, 1));  x_shape=(3,1)  
if(USE_INPUT==2):
    x = np.array([1,2,3,4,5,6]); x = np.reshape(x,(1, 3, 2));  x_shape=(3,2)  
    # x = np.array([1,2,3,4,5,6,7,8,9]); x = np.reshape(x,(1, 3, 3));  x_shape=(3,3)  

#BUILD MODEL 
model = Sequential()
model.add(SimpleRNN(N_OUTPUT, 
    input_shape=x_shape, 
    activation='linear',
    bias_initializer="ones",
    kernel_initializer="ones",
    recurrent_initializer="ones",
    return_sequences=RETURN_SEQUENCES),
    )
if(USE_DENSE): model.add(Dense(units=1, activation='linear'))
# model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

#EXTRACT MATRICES
print("MATRIX SHAPES:")
wx = model.get_weights()[0]; print('wx',wx.shape); #print(wx)
wh = model.get_weights()[1]; print('wh',wh.shape); #print(wh)
bh = model.get_weights()[2]; print('bh',bh.shape); #print(bh)

if(USE_DENSE): 
    wy = model.get_weights()[3]; print('wy',wy.shape); #print(wy)
    by = model.get_weights()[4]; print('by',by.shape); #print(by)

#EVALUTE WITH KERAS
print("x_input",x.shape); print(x)
y_pred_model = model.predict(x)
print("KERAS Prediction:"); print(y_pred_model,y_pred_model.shape)
# print("x[0,0,:] ",x[0,0,:],x[0,0,:].shape)

# #EVALATE MANUALLY (ROW VECTOR REPRESTATION)
h0 = np.zeros(N_OUTPUT).reshape(1,N_OUTPUT);            #print("h0",h0.shape,h0)
h1 = np.matmul(x[0,0,:], wx) + np.matmul(h0,wh) + bh ;  print("h1",h1.shape,h1)
h2 = np.matmul(x[0,1,:], wx) + np.matmul(h1,wh) + bh ;  print("h2",h2.shape,h2)
h3 = np.matmul(x[0,2,:], wx) + np.matmul(h2,wh) + bh ;  print("h3",h3.shape,h3)

if(USE_DENSE): 
    if(RETURN_SEQUENCES):
        print(np.matmul(h1, wy)+by)
        print(np.matmul(h2, wy)+by)
        print(np.matmul(h3, wy)+by)
    else:
        o3 = np.matmul(h3, wy)+by;              print("o3",o3.shape,o3)



# if(USE_INPUT==3):
#     x = np.array([1,2,3,4,5,6,7,8,9,10,11,12]); x = np.reshape(x,(1, 4, 3));  x_shape=(4,3)  

# if(USE_INPUT==3):
#     h4 = np.matmul(x[0,3,:], wx) + np.matmul(h3,wh) + bh ;  print("h4",h4.shape,h4)
# if(USE_DENSE): 
#     if(RETURN_SEQUENCES):
#         print(np.matmul(h1, wy)+by)
#         print(np.matmul(h2, wy)+by)
#         print(np.matmul(h3, wy)+by)

# #EVALATE MANUALLY (COL VECTOR REPRESTATION)
# print("Manual Prediction:")

# #EXTRACT ROWS 
# x1=x[0,0,:].reshape(-1,1); 
# x2=x[0,1,:].reshape(-1,1); 
# x3=x[0,2,:].reshape(-1,1); print(x1.shape)
# bh=bh.reshape(-1,1);

# h0 = np.zeros(N_OUTPUT).reshape(N_OUTPUT,1);            #print("h0",h0.shape,h0)
# h1 = np.matmul(wx, x1) + np.matmul(wh,h0) + bh ;  print("h1",h1.shape); print(h1)
# h2 = np.matmul(wx, x2) + np.matmul(wh,h1) + bh ;  print("h2",h2.shape); print(h2)
# h3 = np.matmul(wx, x3) + np.matmul(wh,h2) + bh ;  print("h3",h3.shape); print(h3)
# if(USE_DENSE): 
#     by=by.reshape(-1,1);
#     o3 = np.matmul(np.transpose(wy),h3)+by;              print("o3",o3.shape,o3)




# model = Sequential()
# model.add(SimpleRNN(6, 
#     input_shape=(8,4), 
#     activation='linear',
#     bias_initializer="ones",
#     kernel_initializer="ones",
#     recurrent_initializer="ones",
#     return_sequences=False),
#     )
# model.summary()
# exit() 