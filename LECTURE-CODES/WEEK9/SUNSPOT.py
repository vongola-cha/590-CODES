


#EXAMPLE MODIFIED FROM: https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras/

from pandas import read_csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN,LSTM,GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#STATMENT OF PROBLEM:
    #GIVEN TIME_STEPS DATA POINTS PREDICT THE NEXT POINT

#USER PARAM
time_steps=10   #given time_steps data points, predict time_steps+1 point
plot_data_partition=False
recurrent_hidden_units=3
epochs=100
f_batch=0.1     #fraction used for batch size
optimizer="RMSprop"
validation_split=0.2

# Parameter split_percent defines the ratio of training examples
def get_train_test(url, split_percent=0.8):
    df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    # Point for splitting data into train and test
    split = int(n*split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data

sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data, test_data, data = get_train_test(sunspots_url)


# PREPARE THE INPUT X AND TARGET Y
def get_XY(dat, time_steps):
    global X_ind,X,Y_ind,Y #use for plotting later

    # INDICES OF TARGET ARRAY
    # Y_ind [  12   24   36   48 ..]; print(np.arange(1,12,1)); exit()
    Y_ind = np.arange(time_steps, len(dat), time_steps); #print(Y_ind); exit()
    Y = dat[Y_ind]

    # PREPARE X
    rows_x = len(Y)
    X_ind=[*range(time_steps*rows_x)]
    del X_ind[::time_steps] #if time_steps=10 remove every 10th entry
    X = dat[X_ind]; 

    #PLOT
    if(plot_data_partition):
        plt.figure(figsize=(15, 6), dpi=80)
        plt.plot(Y_ind, Y,'o',X_ind, X,'.'); plt.show(); exit()    

    #RESHAPE INTO KERAS FORMAT
    X1 = np.reshape(X, (rows_x, time_steps-1, 1))
    # print([*X_ind]); print(X1); print(X1.shape,Y.shape); exit()

    return X1, Y

#PARTITION DATA
testX, testY = get_XY(test_data, time_steps)
trainX, trainY = get_XY(train_data, time_steps)

#CREATE MODEL
model = Sequential()
#COMMENT/UNCOMMENT TO USE RNN, LSTM,GRU
model.add(LSTM(
# model.add(SimpleRNN(
# model.add(GRU(
recurrent_hidden_units,
return_sequences=False,
input_shape=(time_steps-1,1), 
activation='tanh')) 
#NEED TO TAKE THE OUTPUT RNN AND CONVERT TO SCALAR 
model.add(Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.summary()

#TRAIN MODEL
history = model.fit(
trainX, trainY, 
epochs=epochs, 
batch_size=int(f_batch*trainX.shape[0]), 
validation_split=validation_split,
verbose=2)

#HISTORY PLOT
epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.legend()

# MAKE PREDICTIONS
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
# print(trainX.shape, train_predict.shape,testX.shape, test_predict.shape)

#COMPUTE RMSE
train_rmse = np.sqrt(mean_squared_error(trainY, train_predict))
test_rmse = np.sqrt(mean_squared_error(testY, test_predict))
print('Train RMSE: %.3f RMSE' % (train_rmse))
print('Test RMSE: %.3f RMSE' % (test_rmse))    

# PLOT THE RESULT
def plot_result(trainY, testY, train_predict, test_predict):
    plt.figure(figsize=(15, 6), dpi=80)
    #ORIGINAL DATA
    print(X.shape,Y.shape)
    plt.plot(Y_ind, Y,'o', label='target')
    plt.plot(X_ind, X,'.', label='training points');     
    plt.plot(Y_ind, train_predict,'r.', label='prediction');    
    plt.plot(Y_ind, train_predict,'-');    
    plt.legend()
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    plt.show()
plot_result(trainY, testY, train_predict, test_predict)








# #PLOT
# if(True):
#     actual = np.append(trainY, testY)
#     rows = len(actual)
#     plt.figure(figsize=(15, 6), dpi=80)
#     plt.plot(range(rows), actual)
#     start=0; 
#     for i in range(0,trainX.shape[0]):

#         t=[*range(trainX.shape[1])]; #print(t); print(trainX[i,:,0])
#         plt.plot(t, trainX[i,:,0])
#         plt.show()
#         exit()
