from keras.datasets import imdb
from keras import preprocessing
import numpy as np

#DEFINES VOCBULARY TO USE
max_features = 10000   

(x_train, y_train), (x_test, y_test) = imdb.load_data( num_words=max_features)
print(x_train[0][0:10]) # ,y_train.shape)


#TRUNCATE OR PAD (CUTOFF REVIEWS AFTER ONLY 20 WORDS)
maxlen = 20	

#truncating='pre' --> KEEPS THE LAST 20 WORDS
#truncating='post' --> KEEPS THE FIRST 20 WORDS

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,truncating='post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,truncating='post')
print(x_train[0])



#BUILD MODEL
from keras.models import Sequential 
from keras.layers import Flatten, Dense,Embedding

model = Sequential()
#learn 8-dimensional embeddings for each of the 10,000 words
model.add(Embedding(10000, 8, input_length=20))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=1,
batch_size=32, validation_split=0.2)


maxlen = 21

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,truncating='post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,truncating='post')
# print(x_train[0])

model.predict(x_train)

exit()


# print(np.array([1,0,0,0,0,0,0]).reshape(7,1))
# exit()


# test=preprocessing.sequence.pad_sequences([[1, 2, 3], [3, 4, 5, 6], [7, 8]],padding='post', maxlen=3)
# print(test)