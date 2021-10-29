# from matplotlib import pyplot as plt

# #RUN SETUPFILE (SEE SETUP.py)
# from SETUP import *

# #BUILD AND TRAIN MODEL

# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
# model = Sequential() 
# model.add(layers.Embedding(max_features, 32))
# model.add(layers.Bidirectional(layers.LSTM(32)))
# model.add(layers.Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
# history = model.fit(x_train, y_train,
# epochs=10, batch_size=128, validation_split=0.2)



# from PLOT import *




# #BIDIRETIONAL GRU
# # from keras.models import Sequential
# # from keras import layers
# # from keras.optimizers import RMSprop
# # model = Sequential()
# # model.add(layers.Bidirectional(
# #     layers.GRU(32), input_shape=(None, float_data.shape[-1])))
# # model.add(layers.Dense(1))
# # model.compile(optimizer=RMSprop(), loss='mae')
# # history = model.fit_generator(train_gen,
# #                               steps_per_epoch=500,
# #                               epochs=40,
# #                               validation_data=val_gen,
# #                               validation_steps=val_steps)

