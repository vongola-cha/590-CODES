
#MODIFIED FROM: 
# Chollet, Francois. Deep learning with Python. Manning Publications Co., 2018.


#SUPRESS WORNINGS
# import warnings
# warnings.filterwarnings("ignore")

import numpy as np


from tensorflow.keras.datasets import imdb

#GET DATASET AND EXPLORE
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def imbd_explore(indx=0):
    print("----------------------------------")
    #ENTIRE DATA SET
    print("TYPE: ", type(train_data))
    print("TYPE: ", type(train_labels))

    print(train_data.shape) #numpy array of lists
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)
    print(train_data.shape) #numpy array of lists

    #CHECK FOR MAX/MIN
    indx_min=100; indx_max=-10; max_length=-10
    for i in range(0,len(train_data)):
        if(len(train_data[i])>max_length): max_length=len(train_data[i])
        if(max(train_data[i])>indx_max):     indx_max=max(train_data[i])
        if(min(train_data[i])<indx_min):     indx_min=min(train_data[i])
    print("Longest review (num_words):",max_length)
    print("Max train_data:",indx_max)
    print("Min train_data:",indx_min)

    #LENGTH OF FIRST FEW REVIEWS
    for i in range(0,5): print("REVEIW:",i," LENGTH=",len(train_data[i]))
 

    #DATA POINT
    print("TYPE: ",type(train_data[indx]))
    print("TYPE: ",type(train_labels[indx]))

    print(train_data[indx])
    print(train_labels[indx])

    #DECODE
    print("DECODE REVIEW=",indx)
    # word_index is a dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
    #Reverses it, mapping integer indices to words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    #Decodes the review. 
    #indices are offset by 3 because 0, 1, and 2 are reserved 

    decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[indx]])
    print(decoded_review)
    print("----------------------------------")

#INITIAL DATA
# imbd_explore()





#PREPARING THE DATA

# **ONE-HOT Encoding the integer sequences via multi-hot encoding**
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


#COMPARE OLD VS NEW
indx=0
print(x_train[indx].shape)
print(train_data[indx])
print(sorted(train_data[indx]))
print(x_train[indx][0:30])
print(y_train[0:30])

print(x_train.shape)
print(y_train.shape)

#BUILDING YOUR MODEL

from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow.keras import activations

#LOGISTIC REGRESSION MODEL
    #Sequential groups a linear stack of layers into a .keras.Model.

model = keras.Sequential([
layers.Dense(1, activation='sigmoid', input_shape=(10000,)),
])

# LINEAR REGRESSION MODEL
# model = keras.Sequential([
# layers.Dense(1, activation='sigmoid', input_shape=(10000,)),
# ])
print(model.summary())

# HYPERPARAMETERS 
optimizer="rmsprop"
loss_function="mean_squared_error" 
# learning_rate=0.1
numbers_epochs=10
batch_size=512

# COMPILING THE MODEL 
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=["accuracy"])


# DATA PARTITION: BREAK INTO TRAIN+VALIDATION (TEST-->SET ASIDE)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#RESHAPE
partial_y_train=partial_y_train.reshape(len(partial_y_train),1)
y_val=y_val.reshape(len(y_val),1)
y_test=y_test.reshape(len(y_test),1)
# print(partial_x_train.shape,partial_y_train.shape)
# print(x_val.shape,y_val.shape)
# print(x_test.shape,y_test.shape)

# TRAINING YOUR MODEL
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=numbers_epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))

# GET TRAINING OUTPUT
history_dict = history.history
print("HISTORY KEYS: ", history_dict.keys())
print("LOSS RECORD: ", history_dict['loss'])
print("ACCURACY: RECORD", history_dict['accuracy'])

#MAKE PREDICTIONS (OUTPUTS LOSS AND METRICS FOR EACH GROUP)
model.evaluate(partial_x_train, partial_y_train)
model.evaluate(x_val,  y_val)
model.evaluate(x_test, y_test)

#DOUBLE CHECK WHAT KERAS IS DOING INTERNALLY
Y1=partial_y_train  #y_val  
X2=partial_x_train  #x_val 
Y2=model.predict(X2)
print("MSE:",np.mean((Y1-Y2)**2.0))
Y2=np.where(Y2 > 0.5, 1, 0) #a[1, :, None]
AC=1.0-sum(np.absolute(Y1-Y2))/len(Y1)
print("ACC:",AC); #print(Y1.shape,Y2.shape)

#GET MODEL PARAMETERS
weights = model.get_weights() # Getting params

iplot=True
if(iplot):

    import matplotlib.pyplot as plt

    # PLOTTING THE TRAINING AND VALIDATION LOSS 
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    # PLOTTING THE TRAINING AND VALIDATION ACCURACY 
    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()




#XTRA CODE

# print("RESULTS TRAINING:", results1)
# print("RESULTS VALIDATION:", results2)
# print("RESULTS TEST:", results3)

# print("here")
# #PARITY PLOTS
# yp=np.where(model.predict(partial_x_train) > 0.5, 1, 0)
# partial_y_train=partial_y_train.reshape(len(partial_y_train),1)
# print(np.linspace(0.0, 1000.0, num=len(yp)).shape,yp.shape,partial_y_train.shape)
# plt.plot(np.linspace(0.0, 1000.0, num=len(yp)),partial_y_train-yp, "bo", label="Training loss")
# plt.show()
# # plt.clf()
# exit()