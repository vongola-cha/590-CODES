


from keras.models import load_model
from keras.utils.np_utils import to_categorical

model = load_model('model-in.h5',compile=True)

#GET DATA  
from keras.datasets import mnist
(x, y), (x_test, y_test) = mnist.load_data()

#RESHAPE
y=to_categorical(y); y_test=to_categorical(y_test)
def reshape_2(x): return x.reshape((x.shape[0],x.shape[1] * x.shape[2])) 
x = reshape_2(x); x_test = reshape_2(x_test) 
# print(x.shape,y.shape,x_test.shape,y_test.shape)

#EVALUATE LOADED MODEL
print(model.evaluate(x,y,batch_size=len(x)))
print(model.evaluate(x_test,y_test,batch_size=len(x_test)))

#SAVE MODEL
model.save('model-out.h5')   

