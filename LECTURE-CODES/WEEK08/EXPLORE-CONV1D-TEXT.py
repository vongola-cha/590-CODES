import numpy as np

corpus = [
'this movie is garbage',
'this movie is the best thing ive ever seen',
]

def form_dictionary(samples):
    token_index = {};  
    #FORM DICTIONARY WITH WORD INDICE MAPPINGS
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    transformed_text=[]
    for sample in samples:
        tmp=[]
        for word in sample.split():
            tmp.append(token_index[word])
        transformed_text.append(tmp)

    print("CONVERTED TEXT:", transformed_text)
    print("VOCABULARY-2 (SKLEARN): ",token_index)
    return [token_index,transformed_text]

[vocab,x]=form_dictionary(corpus)



# #CHOLLET:  LISTING 6.1 WORD-LEVEL ONE-HOT ENCODING (TOY EXAMPLE)
def one_hot_encode(samples):
    #ONE HOT ENCODE (CONVERT EACH SENTENCE INTO MATRIX)
    max_length = 9
    results = np.zeros(shape=(len(samples),max_length,max(vocab.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = vocab.get(word)
            results[i, j, index] = 1.
    results=results[:,:,1:]
    print("ONE HOT")
    print(results)
    return results

x=one_hot_encode(corpus)

#APPLY CONVOLUTION
import tensorflow as tf

layer1=tf.keras.layers.Conv1D(
	1, 2, activation='relu',
    kernel_initializer="ones",
    bias_initializer="zeros",
	input_shape=x.shape[1:]
	)
y = layer1(x)



def report(x,y,l):
	print("X:"); print(x); 
	print("X SHAPE:",x.shape)
	print("KERNAL SHAPE:",l.get_weights()[0].shape)
	print("KERNAL:"); print(l.get_weights()[0])
	print("Y SHAPE",y.shape)
	print("Y:"); print(y)
report(x,y,layer1)

#GET LAYER INFO
from keras.models import Sequential 
model= Sequential()
model.add(layer1)
model.summary()


#EXPLORE RNN: LISTING 6.21 NUMPY IMPLEMENTATION OF A SIMPLE RNN
timesteps = x.shape[1]
input_features = x.shape[2]
output_features = x.shape[2] #20
print("timesteps,input_features,output_features")
print(timesteps,input_features,output_features)

inputs = x[1]  #np.random.random((timesteps, input_features))


state_t = np.zeros((output_features,)) #INITIALIZE PREVIOUS STATE
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs: #LOOP OVER TIME
	output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
	successive_outputs.append(output_t)
	state_t = output_t

	print("input_t:",input_t)
	print("output_t:",output_t)

final_output_sequence = np.concatenate(successive_outputs, axis=0)
print("FINAL_OUTPUT_SEQUENCE")
print(final_output_sequence)


# print(np.ones(20).reshape(2,10))
# exit()