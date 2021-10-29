from matplotlib import pyplot as plt
import os

data_dir = './jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

#USER INPUT
I_PRINT     = True
I_PLOT      = False
N_col_keep  = 2        #[1,2] #[*range(1,14)]

 #NUMBER ROWS TO KEEP FOR DEBUGGING
N_KEEP      = 1*10000  #1 year=52560    #MAX=420451  

#BOUNDS (B) FOR DATA PARTITION (TRAIN,TEST VAL)
B1=0;       B2=int(0.5*N_KEEP)
B3=B2+1;    B4=int(0.8*N_KEEP)
B5=B4+1;    B6=N_KEEP

#GENERATOR FUNCTION PARAM
lookback    = 1440      #look back 
step        = 1 #6
delay       = 144
batch_size  = 3 #int(N_KEEP*0.01)

#READ DATA FROM CSV
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')

#SPLIT INTO PARTS USING "," DELIMITER
header = lines[0].split(','); header=header[1:1+N_col_keep]
lines = lines[1:N_KEEP+1]

#LOOP OVER LINES AND POPULATE  NP ARRAY
import numpy as np
float_data = np.zeros((len(lines),N_col_keep)) #initialize
for i, line in enumerate(lines):
    #CREATE LIST WITH VALUES FOR ROW (EXCLUDE TIME)
    values = [float(x) for x in line.split(',')[1:1+N_col_keep]]
    #UPDATE ROW
    float_data[i, :] = values


if(I_PLOT):
    temp = float_data[:, float_data.shape[1]]  # temperature (in degrees Celsius)
    plt.plot(range(B1,B2), temp[B1:B2])
    plt.plot(range(B3,B4), temp[B3:B4])
    plt.plot(range(B5,B6), temp[B5:B6])
    plt.show()

    #10 DAYS (10*(24*60/10)=1440)
    plt.plot(range(1440), temp[:1440])
    plt.show()

#NORMALIZE USING TRAINING DATA MEAN/STD (B1 to B2)
mean = float_data[B1:B2].mean(axis=0)
float_data -= mean
std = float_data[B1:B2].std(axis=0)
float_data /= std


#GENERATOR FUNCTION
# data—The original array of floating-point data (normalized)
# lookback—How many timesteps back the input data should go.
# delay—How many timesteps in the future the target should be.
# min_index and max_index—Indices in the data array that delimit which time-
#      steps to draw from. This is useful for keeping a segment of the data for valida-
#      tion and another for testing.
# shuffle—Whether to shuffle the samples or draw them in chronological order.
# batch_size—The number of samples per batch.
# step—The period, in timesteps, at which you sample data. You’ll set it to 6 in
#       order to draw one data point every hour.

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    # print("HERE")
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
            # print(rows); exit()
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        print(rows); # exit()
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        print("here"); # exit()

        yield samples, targets


# instantiate three generators
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=B1,
                      max_index=B2,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=B3,
                    max_index=B4,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=B5,
                     max_index=B6,
                     step=step,
                     batch_size=batch_size)


if(I_PRINT):
    print(header)
    print("BATCHSIZE:",batch_size)
    print("DATA BOUNDS:",B1,B2,B3,B4,B5,B6); # exit()
    print("NUM LINES",len(lines))
    # print("LINE-0",lines[0])
    # print("LAST LINE",lines[-1])
    print("DATA SHAPE:",float_data.shape)
    print("type(train_gen)",type(train_gen))

# VISUALIZE GENERATOR
I_PLOT      = True

if(I_PLOT):
    plt.figure() #INITIALIZE FIGURE 
    plt.plot(range(B1,B2), float_data[B1:B2, float_data.shape[1]-1],'bo')

    for i in range(0,10):

        samples, targets = next(train_gen)
        print(type(samples),type(targets))
        print(samples.shape,targets.shape)      

        plt.plot(range(0,len(samples[0,:,float_data.shape[1]-1])), samples[0,:, float_data.shape[1]-1],'ro')
        plt.pause(0.1)


    plt.show()


    # N=sum(1 for dummy in train_gen)
    # print(N)

        # exit(); #NEED TO EXIT 

exit()




val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)

# def evaluate_naive_method():
#     batch_maes = []
#     for step in range(val_steps):
#         samples, targets = next(val_gen)

#         print(type(samples),type(targets))
#         print(samples.shape,targets.shape)
#         preds = samples[:, -1, 1]
#         mae = np.mean(np.abs(preds - targets))
#         batch_maes.append(mae)
#         print(step); exit()
#     print(np.mean(batch_maes))
# evaluate_naive_method()

exit()

 # celsius_mae = 0.29 * std[1]

# print("here"); exit()

#----------------------------------------------
#XTRA
#----------------------------------------------

# print(np.array(lines))
# print(float_data.shape)
# exit()

# #PLOT TEMPERATURES (FULL AND FIRST 10 DAYS)
# from matplotlib import pyplot as plt
# temp = float_data[:, 1]   
# plt.plot(range(len(temp)), temp); plt.show()
# plt.plot(range(1440), temp[:1440]); plt.show()


# #NORMALIZE
# float_data=(float_data-np.mean(float_data,axis=0))/np.std(float_data,axis=0)


# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
# model = Sequential()
# model.add(layers.Embedding(max_features, 128, input_length=max_len))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.MaxPooling1D(5))
# model.add(layers.Conv1D(32, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# model.summary()
# model.compile(optimizer=RMSprop(lr=1e-4),
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)



# exit()
# for i, line in enumerate(lines):
#     parts=line.split(',')
#     # date=
#     M=float(parts[0].split('.')[0])
#     D=float(parts[0].split('.')[1])
#     Y= parts[0].split('.')[2].split(' ')[0]
#     h= parts[0].split('.')[2].split(' ')[1].split(':')[0]
#     m= parts[0].split('.')[2].split(' ')[1].split(':')[1]
#     s= parts[0].split('.')[2].split(' ')[1].split(':')[2]

#     print(i,parts[0])
#     print("M",M,D,Y,h,m,s)

   
#     if(i>10000): exit()
# exit()
