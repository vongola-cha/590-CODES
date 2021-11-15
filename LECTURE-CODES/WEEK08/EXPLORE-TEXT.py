
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
'the dog ate my homework',
'the homework ate my dog',
'ate my dog',
'the dog ate my dog'
]

#-----------------------------
#BASIC TOKENIZATION EXAMPLE
#-----------------------------

A=corpus[0]
print("WORD TOKENS",A.split())

temp=[]
for char in A: temp.append(char)

print("CHARACTER TOKENS",temp)



#-----------------------------
#DOCUMENT TERM MATRIX
#-----------------------------
#FREQUENCY COUNT MATRIX

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus) #WILL EXCLUDE WORDS OF LENGTH=1
print("VOCABULARY-1",vectorizer.get_feature_names())
print("DOCUMENT TERM MATRIX")
print(X.toarray())

#-----------------------------
#FORM DICTIONARY AND ENCODE AS INDEX TOKENS
#-----------------------------

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


#-----------------------------
#VECTORIZE
#-----------------------------

#CHOLLET; IMDB (CHAPTER-3: P69)
def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

print(x[0]); #print(x); print(len(x[0]))
x = vectorize_sequences(x,10)
print(x); #print(x); print(x.shape)


# #CHOLLET:  LISTING 6.1 WORD-LEVEL ONE-HOT ENCODING (TOY EXAMPLE)
def one_hot_encode(samples):
    #ONE HOT ENCODE (CONVERT EACH SENTENCE INTO MATRIX)
    max_length = 10
    results = np.zeros(shape=(len(samples),max_length,max(vocab.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = vocab.get(word)
            results[i, j, index] = 1.
    print("ONE HOT")
    print(results)

one_hot_encode(corpus)


#KERAS ONEHOT ENCODING
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
one_hot_results = tokenizer.texts_to_matrix(corpus, mode='binary')
word_index = tokenizer.word_index
print("KERAS")
print(sequences)
print(one_hot_results)
print('Found %s unique tokens.' % len(word_index))


# ONE-HOT HASHING TRICK,
dimensionality = 10
max_length = 10
results = np.zeros((len(corpus), max_length, dimensionality))
for i, sample in enumerate(corpus):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

print("HASHING")
print(results)



#CHOLLET: LISTING 6.2 CHARACTER-LEVEL ONE-HOT ENCODING (TOY EXAMPLE)
# import string
# characters = string.printable; # print(characters) #, len(characters))
# token_index = dict(zip(range(1, len(characters) + 1), characters))
# max_length = 50
# results = np.zeros((len(corpus), max_length, max(token_index.keys()) + 1))
# for i, sample in enumerate(corpus):
#     for j, character in enumerate(corpus):
#         index = token_index.get(character)
#         results[i, j, index] = 1.
# print(results.shape)
# print(results[1].shape)
# print(results)

