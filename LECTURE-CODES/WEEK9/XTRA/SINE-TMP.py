import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#-----------------------------------
#USER PARAM
#-----------------------------------

function=1     #1,2 (differnt funtion choices)
I_SQUARE=True
N=200           #NUMBER OF DATA POINTS
fmin=1          #MIN FREQUENCY

#GENERATOR FUNCTION PARAM
lookback            = 10         #number of timestep backwards to use
step                = 2          #use every data point 
delay               = 5          #number of timesteps in future
I_PLOT_1            = False
I_PLOT_GENERATOR    = True

#-----------------------------------
#BUILD DATASET
#-----------------------------------
t=np.linspace(0,2.25/fmin,N).reshape(N,1)

#SIMPLE SINE FUNCTION
if(function==1): 
    y=10*np.sin(2*np.pi*fmin*t) 

#FOURIER SERIES
if(function==2): 
    #GENERATER FOURIER SERIES WITH RANDOME AMPLITURES
    y=0;
    for f in range(fmin,5*fmin):
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
B1=0;       B2=int(0.5*N);   
B3=B2;      B4=int(1.0*N)

#OPTIMIZER PARAM
EPOCHS           = 100
batch_size       = 2 #int(1.0*B2)  #FULL BATCH
steps_per_epoch  = np.ceil(B2/batch_size)


print("BOUNDS",B1,B2,B3,B4); # exit()
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
            rows = np.arange(i-1, min(i + batch_size -1, max_index-delay))
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
        print("batch=",rows); print("(batchsize,lookback,step,delay)=",batch_size,lookback,step,delay)
        for j, row in enumerate(rows):
            # print(j,row)
            indices = range(rows[j] - lookback+1, rows[j]+1, step)
            indices_global.append([*indices])
            print("sample:",rows[j],[*indices],rows[j] + delay); #exit()
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][data.shape[-1]-1]

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
                      batch_size=batch_size)
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


    #CALL GENERATOR-5 TIMES 
    for i in range(0,int(B2/2)):  
        samples, targets  = next(train_gen)
        print("sample shape=",samples.shape,"target shape=",targets.shape) #,len(indices_global))

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


