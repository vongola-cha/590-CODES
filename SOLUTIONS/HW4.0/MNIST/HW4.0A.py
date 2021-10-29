


# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
from keras import callbacks

from keras.utils.np_utils import to_categorical
from keras.datasets import reuters, imdb, boston_housing
import matplotlib.pyplot as plt
import json

#-------------------------
#READ PARAM 
#-------------------------
with open('param.json') as f:
    param = json.load(f)  #read into dictionary

#ADD PARAM DICTIONARY INTO GLOBAL VARIABLE DICTIONARY
globals().update(param);  #print(globals())

#-------------------------
#GET DATA AND DEFINE PARAM
#-------------------------

#ADD CHANNELS DIMENSION (RANK-3 --> RANK-4)
def reshape_1(x): return x.reshape((x.shape[0],x.shape[1],x.shape[1],1))

#GET DATA
if(DATASET=='MNIST'): 
    from keras.datasets import mnist
    (x, y), (x_test, y_test) = mnist.load_data()
    x=reshape_1(x); x_test=reshape_1(x_test)

if(DATASET=='MNIST-F'): 
    from keras.datasets import fashion_mnist
    (x, y), (x_test, y_test) = fashion_mnist.load_data()
    x=reshape_1(x); x_test=reshape_1(x_test)

if(DATASET=='CIFAR-10'): 
    from keras.datasets import cifar10
    (x, y), (x_test, y_test) = cifar10.load_data()

#SHOW RANDOM IMAGE
if(SHOW_IMAGE):
    rand_ID=int(np.random.uniform(0,x.shape[0],1)[0]); #print(rand_ID)
    plt.imshow(x[rand_ID]); plt.show()

#DOWNSIZE DATASET FOR DEBUGGING IF DESIRED
if(FRAC_KEEP<1.0):
    NKEEP=int(FRAC_KEEP*x.shape[0])
    rand_indices = np.random.permutation(x.shape[0])
    x=x[rand_indices[0:NKEEP]]
    y=y[rand_indices[0:NKEEP]]

def explore(x,y,tag=''):
    print("------EXPLORE RAW "+tag.upper()+" DATA------")
    print("x shape:",x.shape)
    print("x type:",type(x),type(x[0]))  
    print("y shape:",y.shape)
    print("y type:",type(y),type(y[0]))
    for i in range(0,5):
        if(str(type(x[i]))=="<class 'numpy.ndarray'>"):
            print(" x["+str(i)+"] shape:",x[i].shape, "y["+str(i)+"]=",y[i]) 
        if(str(type(x[i]))=="<class 'list'>"):
            print(" x["+str(i)+"] len:",len(x[i]), "y["+str(i)+"]=",y[i]) 

explore(x,y,"TRAIN")
explore(x_test,y_test,"TEST")


#-------------------------
#DATA PREPROCESSING/NORM 
#-------------------------

#TAKE MEAN AND STD DOWN COLUMNS (I.E DOWN SAMPLE DIMENSION)
XMEAN=np.mean(x,axis=0); XSTD=np.std(x,axis=0); #print(XMEAN,XSTD) 
YMEAN=np.mean(y,axis=0); YSTD=np.std(y,axis=0); #print(YMEAN,YSTD)

if(X_NORM=="ZSCORE"): x=(x-XMEAN)/XSTD;  x_test=(x_test-XMEAN)/XSTD;            
if(Y_NORM=="ZSCORE"): y=(y-YMEAN)/YSTD;  y_test=(y_test-YMEAN)/YSTD;            

if(X_NORM=="MAX"): x=x/np.max(x);   x_test=x_test/np.max(x_test);
if(Y_NORM=="MAX"): y=y/np.max(y);   y_test=y_test/np.max(y_test)

if(Y_ENCODE=="BINARY"): 
    y = np.asarray(y).astype("float32")
    y_test = np.asarray(y_test).astype("float32")

if(Y_ENCODE=="ONEHOT" or X_ENCODE=="ONEHOT"): # 3 --> [0,0,0,1,0,0, ...]
    from keras.utils.np_utils import to_categorical
    if(X_ENCODE=="ONEHOT"): x=to_categorical(x); x_test=to_categorical(x_test)
    if(Y_ENCODE=="ONEHOT"): y=to_categorical(y); y_test=to_categorical(y_test)

def reshape_2(x): return x.reshape((x.shape[0],x.shape[1] * x.shape[2] * x.shape[3])) 
if(MODEL_TYPE=="DFF"): #UNWRAP (A,B,X,D) MATRICES INTO LONG VECTORS (A,B*C*D)) 
    x = reshape_2(x); x_test = reshape_2(x_test) 

explore(x,y,"TRAIN")
explore(x_test,y_test,"TEST")

#-----------------
#MODEL
#-----------------

#BUILD KERAS MODEL
def build_model():

    # #BUILD LAYER ARRAYS FOR DENSE PART OF ANN
    # ACTIVATIONS=[]; LAYERS=[]   
    # for i in range(0,N_DENSE_LAYER):
    #     LAYERS.append(N_DENSE_NODES)
    #     ACTIVATIONS.append(ACT_TYPE)
    # print("LAYERS:",LAYERS)
    # print("ACTIVATIONS:", ACTIVATIONS)

    model = models.Sequential()
    
    #FIRST LAYER
    if(MODEL_TYPE=="CNN"):
        model.add(layers.Conv2D(N_FILTER1, kernel_size=KERN_SIZE, activation=ACT_TYPE, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
        model.add(layers.MaxPooling2D((POOL_SIZE, POOL_SIZE)))

    #ADD CONV LAYERS AS IF NEEDED:
    if(MODEL_TYPE=="CNN"):
        for i in range(1,N_COV2D_LAYER-1):
            model.add(layers.Conv2D(N_FILTER2, kernel_size=KERN_SIZE, activation=ACT_TYPE)) 
            model.add(layers.MaxPooling2D((POOL_SIZE, POOL_SIZE)))
        model.add(layers.Conv2D(N_FILTER2, kernel_size=KERN_SIZE, activation=ACT_TYPE)) 
        model.add(layers.Flatten())
    
    #ADD DENSE LAYERS AS NEEDED 
    for i in range(0,N_DENSE_LAYER):
        if(MODEL_TYPE=="DFF" and i==0):
            model.add(layers.Dense(N_DENSE_NODES, activation=ACT_TYPE, input_shape=(x_train.shape[1],), kernel_regularizer=regularizers.l2(L2_CONSTANT)))
        else:
            model.add(layers.Dense(N_DENSE_NODES, activation=ACT_TYPE, kernel_regularizer=regularizers.l2(L2_CONSTANT)))

    #OUTPUT LAYER
    model.add(layers.Dense(y.shape[1], activation=OUT_ACT, kernel_regularizer=regularizers.l2(L2_CONSTANT)))
    #print(model.summary()); exit()

    #COMPILE
    if(OPTIMIZER=='rmsprop' and LR!=0):
        opt = optimizers.RMSprop(learning_rate=LR)
    else:
        opt = OPTIMIZER

    model.compile(
    optimizer=opt, 
    loss=LOSS, 
    metrics=METRICS
                 )
    return model 

#-----------------
#TRAIN MODEL
#-----------------

#AVERAGE WITH NEIGHBORS
def smooth(x):
    x=np.array(x); 
    xp1=x[2:]; xc=x[1:(x.shape[0]-1)]; xm1=x[0:(x.shape[0]-2)]
    return (xm1+xc+xp1)/3.0

samples_per_k = x.shape[0] // N_KFOLD
if(N_KFOLD==1): samples_per_k=int(0.2*x.shape[0])
print(N_KFOLD,samples_per_k,x.shape,y.shape)

#RANDOMIZE ARRAYS
rand_indx = np.random.permutation(x.shape[0])
x=x[rand_indx]; y=y[rand_indx]

#LOOP OVER K FOR KFOLD
for k in range(0,N_KFOLD):

    print('---PROCESSING FOLD #', k,"----")
    
    #VALIDATION: (SLIDING WINDOW LEFT TO RIGHT)
    x_val = x[k * samples_per_k: (k + 1) * samples_per_k]
    y_val = y[k * samples_per_k: (k + 1) * samples_per_k]

    #TRAINING: TWO WINDOWS (LEFT) <--VAL--> (RIGHT)
    x_train = np.concatenate(
        [x[:k * samples_per_k],
         x[(k + 1) * samples_per_k:]],
        axis=0)
    
    y_train = np.concatenate(
        [y[:k * samples_per_k],
         y[(k + 1) * samples_per_k:]],
        axis=0)

    #PRINT TO SEE WHAT LOOP IS DOING
    print("A",k * samples_per_k,(k + 1) * samples_per_k)
    print("B",x[:k * samples_per_k].shape, x[(k + 1) * samples_per_k:].shape)
    print("C",x_train.shape,y_train.shape)
    print("D",x_val.shape,y_val.shape)
    BATCH_SIZE=int(FRAC_BATCH*x_train.shape[0])

    #BUILD MODEL 
    model = build_model()
    if(k==0):  model.summary()

    #DEFINE CHECKPOING 
    model_checkpoint_callback= callbacks.ModelCheckpoint(
    'OPT-MODEL.h5',
    monitor="val_loss",
    verbose=1,
    save_best_only=True)

    #FIT MODEL
    history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=VERSBOSE,
    callbacks=[model_checkpoint_callback],
    validation_data=(x_val, y_val)
    )


    #DETERMINE MINIMIAL LOSS 
    loss = smooth(history.history['loss'])
    val_loss = smooth(history.history['val_loss'])
    MT = smooth(history.history[METRICS[0]])
    MV = smooth(history.history['val_'+METRICS[0]])
    epochs = np.array(range(0, len(loss)));
    epoch_opt=epochs[val_loss==np.min(val_loss)][0]

    #BASIC PLOTTING              
    if(k==N_KFOLD-1 and I_PLOT):
        plt.plot(epoch_opt*np.ones(len(loss)),loss, '-', label='Training loss')
        plt.plot(epochs, loss, 'bo-', label='Training loss')
        plt.plot(epochs, val_loss, 'g*-', label='Validation loss')
        plt.plot(epochs, np.absolute(val_loss-loss), 'ro-',  label='|Training-Validation| loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('LOSS.png'); # plt.show()

        #METRICS
        if(len(METRICS)>0):
            for metric in METRICS:
                plt.clf()
                # plt.plot(MV, MT, 'bo', label='Training vs Val '+metric)
                plt.plot(epochs, MT, 'b-o', label='Training '+metric)
                plt.plot(epoch_opt*np.ones(len(loss)), MT, '-',  label='Validation '+metric)
                plt.plot(epochs, MV[epoch_opt]*np.ones(len(loss)), '-', label='Training '+metric)
                plt.plot(epochs, MV, 'r-*',  label='Validation '+metric)
                ## plt.plot(epochs, np.absolute(MT-MV), 'ro',  label='|Training-Validation| '+metric)
                plt.title('Training and validation '+metric)
                plt.xlabel('Epochs')
                plt.ylabel(metric)
                plt.legend()
                plt.savefig(metric+'.png'); # plt.show()


    train_values  = model.evaluate(x_train, y_train,batch_size=y_val.shape[0],verbose=VERSBOSE)
    val_values   = model.evaluate(x_val,   y_val,batch_size=y_val.shape[0],verbose=VERSBOSE)
    # scores.append(val_mae)
    print("--------------------------")
    print("RESULTS FOR K=",k)
    print("TRAIN:",train_values)
    print("VALIDATION:",val_values)
    print("--------------------------")

    if(k==0):
        train_ave=np.array(train_values)
        val_ave=np.array(val_values)
    else:
        train_ave=train_ave+np.array(train_values)
        val_ave=val_ave+np.array(val_values)

#AVERAGE
train_ave=train_ave/N_KFOLD
val_ave=val_ave/N_KFOLD
test_values = model.evaluate(x_test, y_test,batch_size=y_val.shape[0],verbose=0)

#FINAL REPORT
print("--------------------------")
print("epoch_opt:",epoch_opt+2)
print("Train_opt:",MT[epoch_opt])
print("Val_opt:",MV[epoch_opt])
print("AVE TRAIN:",train_ave)
print("AVE VALIDATION:",val_ave)
print("TEST:",test_values)
# print("VAL  RATIOS",np.array(train_ave)/np.array(val_ave))
# print("TEST RATIOS",np.array(train_values)/np.array(test_values))
print("--------------------------")

from keras.models import load_model
model.save('FINAL-MODEL.h5')   




#-------------------------
#PARAMETER MEANINGS
#-------------------------
#STORED IN JSON FILE

# DATASET     =   'MNIST-F'     #FASHION   

# #HIDDEN LAYER PARAM
# N_HIDDEN    =   2           #NUMBER OF HIDDLE LAYERS
# N_NODES   =   64          #NODES PER HIDDEN LAYER
# ACT_TYPE    =   'relu'      #ACTIVATION FUNTION FOR HIDDEN LAYERS

# #TRAINING PARAM
# FRAC_KEEP   =   1.0         #SCALE DOWN DATASET FOR DEBUGGGING 
# FRAC_BATCH  =   0.01        #CONTROLS BATCH SIZE
# OPTIMIZER =   'rmsprop' 
# LR          =   0           #ONLY HAS EFFECT WHEN OPTIMIZER='rmsprop' (IF 0 USE DEFAULT)
# L2_CONSTANT =   0.0         #IF 0 --> NO REGULARIZATION (DEFAULT 1e-4)

# EPOCHS        =   10
# N_KFOLD     =   1           #NUM K FOR KFOLD (MAKE 1 FOR SINGLE TRAINING RUN)
# VERSBOSE    =   1
# NORM        = "NONE"




#TRAINING PARAM




# model = models.Sequential()
# model.add(
# layers.Conv2D(
# filters=100, 
# kernel_size=5, 
# activation='relu', 
# padding="same",
# input_shape=(300, 300, 3)))

# model.summary()

# exit()
