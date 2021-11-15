
#https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e

# pip3 install keras
# pip3 install ann_visualizer
# pip install graphviz

# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D
import numpy

model = Sequential()
model.add(Dense(6, input_dim=6, activation='relu'))
# model.add(Dense(6, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
from ann_visualizer.visualize import ann_viz;

ann_viz(model, title="My first neural network")