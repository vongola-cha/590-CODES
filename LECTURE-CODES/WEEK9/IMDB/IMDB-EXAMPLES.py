



from keras.datasets import imdb
from keras import preprocessing
import numpy as np
from keras.models import Sequential 
from keras import layers
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop

#---------------------------
#USER PARAM
#---------------------------
max_features = 10000    #DEFINES SIZE OF VOCBULARY TO USE
maxlen       = 250      #CUTOFF REVIEWS maxlen 20 WORDS)
epochs       = 8
batch_size   = 1000
verbose      = 1
embed_dim    = 8        #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
lr           = 0.001    #LEARNING RATE

#---------------------------
#GET AND SETUP DATA
#---------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train[0][0:10]) # ,y_train.shape)

#truncating='pre' --> KEEPS THE LAST 20 WORDS
#truncating='post' --> KEEPS THE FIRST 20 WORDS
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,truncating='post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,truncating='post')
# print('input_train shape:', x_train.shape)
print(x_train[0][0:10]) # ,y_train.shape)
# print('input_train shape:', x_train.shape)

#PARTITION DATA
rand_indices = np.random.permutation(x_train.shape[0])
CUT=int(0.8*x_train.shape[0]); 
train_idx, val_idx = rand_indices[:CUT], rand_indices[CUT:]
x_val=x_train[val_idx]; y_val=y_train[val_idx]
x_train=x_train[train_idx]; y_train=y_train[train_idx]
print('input_train shape:', x_train.shape)


#---------------------------
#plotting function
#---------------------------
def report(history,title='',I_PLOT=True):

    print(title+": TEST METRIC (loss,accuracy):",model.evaluate(x_test,y_test,batch_size=50000,verbose=verbose))

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

        plt.title(title)
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.close()

print("---------------------------")
print("DFF (MLP)")  
print("---------------------------")

model = Sequential()
#learn 8-dimensional embeddings for each of the 10,000 words
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="DFF")


print("---------------------------")
print("SimpleRNN")  
print("---------------------------")

model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.SimpleRNN(32)) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="SimpleRNN")


print("---------------------------")
print("LSTM")  
print("---------------------------")

model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.LSTM(32)) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="LTSM")


print("---------------------------")
print("LSTM-BIDIRECTIONAL")  
print("---------------------------")

model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Bidirectional(layers.LSTM(32))) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="LSTM-BIDIRECTIONAL")


print("---------------------------")
print("GRU")  
print("---------------------------")

model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.GRU(32)) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="GRU")


print("---------------------------")
print("1D-CNN")  
print("---------------------------")

model = Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 

model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="CNN")


print("---------------------------")
print("1D-CNN+GRU")  
print("---------------------------")

model = Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32)) #, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="CNN_TO_RNN")


exit()

print("---------------------------")
print("LSTM-REVERSE")  
print("---------------------------")

#RELOAD DATA
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

#REVERSE ORDER OF DATA
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train =  preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test =  preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_split=0.2,verbose=verbose)
report(history,title="GRU")


exit()









exit()


# print(np.array([1,0,0,0,0,0,0]).reshape(7,1))
# exit()


# test=preprocessing.sequence.pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]],padding='post', maxlen=3)
# print(test)












# from keras.datasets import imdb
# from keras.preprocessing import sequence
# from keras import layers
# from keras.models import Sequential


# max_features = 10000
# maxlen = 500

# (x_train, y_train), (x_test, y_test) = imdb.load_data(
#     num_words=max_features)


# def report():




  
# #NO BIDIRECTIONS






# exit()

# #REVERSE
# x_train = [x[::-1] for x in x_train]
# x_test = [x[::-1] for x in x_test]


# #NO BIDIRECTIONS
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# model = Sequential()
# model.add(layers.Embedding(max_features, 128))
# model.add(layers.LSTM(32))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])

# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)

# #BI-DIRECTIONSAL 
# model = Sequential() 
# model.add(layers.Embedding(max_features, 32)) 
# model.add(layers.Bidirectional(layers.LSTM(32))) 
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
# history = model.fit(x_train, y_train,
# epochs=10, batch_size=128, validation_split=0.2)


# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
# model = Sequential()
# model.add(layers.Bidirectional(
#     layers.GRU(32), input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)
