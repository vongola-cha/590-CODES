import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#-----------------------------------
#USER PARAM
#-----------------------------------

function        = 2      #1,2 (differnt funtion choices)
I_SQUARE        = False  #GENERATE TWO FEATURE
N               = 200    #NUMBER OF DATA POINTS
fmin            = 1      #MIN FREQUENCY
EPOCHS          = 100

#GENERATOR FUNCTION PARAM
lookback            = 20         #number of timestep backwards to use
step                = 1          #use every data point 
delay               = 10          #number of timesteps in future
I_PLOT_1            = False
I_PLOT_GENERATOR    = False
f_batch             = 0.01
lr                  = 0.001

#-----------------------------------
#BUILD DATASET
#-----------------------------------
t=np.linspace(0,4.25/fmin,N).reshape(N,1)

#SIMPLE SINE FUNCTION
if(function==1): 
    y=10*np.sin(2*np.pi*fmin*t) 

#FOURIER SERIES
if(function==2): 
    #GENERATER FOURIER SERIES WITH RANDOME AMPLITURES
    y=0;
    for f in range(fmin,10*fmin):
        Af=np.random.uniform(1,5,1)[0]
        Bf=np.random.uniform(1,5,1)[0]
        y=y+Af*np.sin(2*np.pi*f*t)+Bf*np.cos(2*np.pi*f*t)
 
#CREATE TWO FEATURES (y,y^2)
if(I_SQUARE):
    y=np.concatenate((y,y**2),axis=1)

float_data=y; #DEFINE TO BE CONSISTANT WITH TEXTBOOK

#-----------------------------------
#PARTITION DATASET
#-----------------------------------

#TRAIN/VALIDATE SPLIT
B1=0;              B2=int(0.5*N);   
B3=B2;             B4=int(1.0*N)
batch_size         = max(1,int(f_batch*B2))   
batch_size_v       = max(1,int(f_batch*(B4-B3)))  

steps_per_epoch  = np.ceil(B2/batch_size)

print("BOUNDS",B1,B2,B3,B4)
print("SHAPES",t.shape,float_data.shape)

#NORMALIZE USING TRAINING DATA MEAN/STD (B1 to B2)
mean = float_data[B1:B2].mean(axis=0)
float_data -= mean
std = float_data[B1:B2].std(axis=0)
float_data /= std

#PLOT ORIGINAL SIGNAL
if(I_PLOT_1):
    plt.figure();plt.plot(t, y[:,0] , 'r-' );
    if(I_SQUARE): plt.plot(t, y[:,1] , 'b-' );
    plt.show()

#-----------------------------------
#GENERATOR FUNCTION
#-----------------------------------

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
#NOTE: lookback/step must be an integer


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):

    global indices_global

    #BEFORE WHILE IS ONLY EVALUATED ON FIRST PASS
    #FUTURE CALLS JUST ITERATE THE WHILE LOOP
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback; print(min_index,i)
    if(lookback % step !=0): print("ERROR: lookback not divisble by step !=0"); exit()
    # print("B",i,min_index,max_index,lookback,delay,step,batch_size); #exit()
    while 1: #INFINITE LOOP

        #GENERATE "batch_size" STARTING POINTS (ONE FOR EACH BATCH)

        #RANDOM STARTING POINTS
        if shuffle:
            rows = np.random.randint(
                min_index + lookback,
                max_index -  delay, size=batch_size)
         #CHRONOLOGICAL
        else: 
            #RESET AT END 
            if i + delay >= max_index:
                i = min_index + lookback #loop back to beginiing 

            #batch_size=3 i=73 --> rows=[73 74 75]    
            rows = np.arange(i-1, min(i + batch_size-1, max_index-delay))
            #INCREMENT FIRST STARTING POINT
            i += len(rows)
        #print("A",i,rows);# exit()

        #INIITIALIZE ARRAY TO STORE SAMPLES AND TARGETS
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))

        #FILL ARRAYS
        indices_global=[]; 
        #print("batch=",rows); print("(batchsize,lookback,step,delay)=",batch_size,lookback,step,delay)
        for j, row in enumerate(rows):
            # print(j,row)
            indices = [*range(rows[j] - lookback+1, rows[j]+1)]#, step)
            indices.reverse(); indices = indices[::step]; indices.reverse()
            indices_global.append(indices)
            #print("sample:",rows[j],indices,rows[j] + delay); #exit()
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][data.shape[-1]-1]
        #exit()

        yield samples, targets

#-----------------------------------
# INSTANTIATE GENERATORS
#-----------------------------------
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=B1,
                      max_index=B2,
                      shuffle=False,
                      step=step,
                      batch_size=batch_size)


val_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=B3,
                      max_index=B4,
                      shuffle=False,
                      step=step,
                      batch_size=batch_size_v)
val_steps = (B4 - B3 - lookback)


#-----------------------------------
# VISUALIZE GENERATOR
#-----------------------------------
N_FEATURES=float_data.shape[-1];  #print(N_FEATURES)
if(I_PLOT_GENERATOR and N_FEATURES<3):

    # #INITIALIZE PLOT
    plt.ion()  #Enable interactive mode.
    fig,ax = plt.subplots(N_FEATURES+1,1,figsize=(15,15))
    for i in range(0,N_FEATURES):
        ax[i].plot(t[range(B1,B2)], float_data[B1:B2,i],'bo')
        ax[i].plot(t[range(B3,B4)], float_data[B3:B4,i],'go')

    #CALL GENERATOR-10 TIMES 
    # print(batch_size,B1)
    for i in range(0,int(2*B2/batch_size)):  
        samples, targets  = next(train_gen)
        #print("samples.shape",samples.shape,"targets.shape",targets.shape) #,len(indices_global))

        #LOOP OVER DATA POINTS IN BATCH
        for point in range(0,len(indices_global)):
            # print("indices:", indices_global[point])
            #LOOP OVER FEATUERS
            for feature in range(0,N_FEATURES):
                ax[feature].plot(t[indices_global[point]], samples[point,:,feature],'o',linewidth=5)
                #ax[feature].plot(t[indices_global[point]], samples[point,:,feature],linewidth=5)

            #TARGETS ONLY MAP TO LAST NUMBER IN ARRAY
            ax[-2].plot(t[max(indices_global[point])+delay], targets[point],'o', markersize=13)
            plt.draw()
            plt.pause(0.1)

        #REPLOT
        for i in range(0,N_FEATURES):
            ax[i].clear()
            ax[i].plot(t[range(B1,B2)], float_data[B1:B2,i],'bo')
            ax[i].plot(t[range(B3,B4)], float_data[B3:B4,i],'go')

    exit()



#-----------------------------------
#TRAINING 
#-----------------------------------

def plot1(history,title="ANN"):
    #PLOT HISTORY
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
    plt.close()

#PLOT TEST DATA
def plot2(model,title="ANN"):
    t1=[]; yv1=[]; yv2=[]

    for i1 in range(0,int((B4-B3)/batch_size_v)):
        samples, targets  = next(val_gen)
        #print(samples.shape,targets.shape,len(indices_global),model.predict(samples).shape)
        for i in range(0,len(indices_global)):
            yv2.append(model.predict(samples)[i,0])
            yv1.append(targets[i])
            t1.append(t[max(indices_global[i])+delay][0])
            #print(t1[i],yv1[i],yv2[i])

    plt.plot(t1, yv1, 'bo', label='test: exact')
    plt.plot(t1, yv2, 'ro', label='predicted')

    plt.plot(t[range(B3,B4)], float_data[B3:B4,-1],'r-', label='')
    plt.title('predictions')
    plt.legend()
    plt.show()

#-----------------------------------
#TRAINING
#-----------------------------------

input_shape=(lookback // step, float_data.shape[-1])


print("---------------------------")
print("DFF (MLP)")  
print("---------------------------")

model = Sequential()
model.add(layers.Flatten(input_shape=input_shape)) 
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(lr=lr), loss='mae') 
model.summary()

history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=val_gen, validation_steps=val_steps)
# model.save('model-DFF.h5')   
plot1(history,"DFF")
plot2(model,"DFF")


print("---------------------------")
print("SimpleRNN")  
print("---------------------------")

model = Sequential() 
model.add(layers.SimpleRNN(32,activation='relu',input_shape=input_shape)) 
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=RMSprop(lr=lr), loss='mae') 
model.summary()
history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=val_gen, validation_steps=val_steps)
plot1(history,"DFF")
plot2(model,"DFF")


print("---------------------------")
print("LSTM")  
print("---------------------------")

model = Sequential() 
model.add(layers.LSTM(32,activation='tanh',input_shape=input_shape)) 
model.add(layers.Dense(1, activation='linear'))
model.compile(optimizer=RMSprop(), loss='mae') 
model.summary()
history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, validation_data=val_gen, validation_steps=val_steps)
plot1(history,"DFF")
plot2(model,"DFF")



exit()









# #MAKE PREDICTIONS
# if(batch_size==1): #STOCASTIC
#     t1=[]; yv1=[]; yv2=[]
#     for i in range(B3,B4):
#         samples, targets  = next(val_gen)
#         yv2.append(model.predict(samples)[0,0])
#         yv1.append(targets[0])
#         t1.append(t[max(indices_global[0])+delay+1])
#         # print(i,max(indices_global[0])+delay+1,model.predict(samples)[0,0],targets[0])
# if(batch_size!=1): #BATCH
#     t1=[]; yv1=[]; yv2=[]
#     samples, targets  = next(val_gen)
#     #print(samples.shape,targets.shape,len(indices_global),model.predict(samples).shape)
#     for i in range(0,len(indices_global)):
#         yv2.append(model.predict(samples)[i,0])
#         yv1.append(targets[i])
#         t1.append(t[max(indices_global[i])+delay+1][0])
#         #print(t1[i],yv1[i],yv2[i])

#     samples, targets  = next(train_gen)
#     # print(samples.shape,targets.shape,len(indices_global),model.predict(samples).shape)
#     for i in range(0,len(indices_global)):
#         yv2.append(model.predict(samples)[i,0])
#         yv1.append(targets[i])
#         t1.append(t[max(indices_global[i])+delay+1][0])
#         #print(t1[i],yv1[i],yv2[i])



