
#MODIFIED FROM 
#https://pytorch.org/tutorials/beginner/examples_tensor/polynomial_numpy.html#sphx-glr-beginner-examples-tensor-polynomial-numpy-py

import numpy as np
import math
import matplotlib.pyplot as plt


# CREATE RANDOM INPUT AND OUTPUT DATA
x = np.linspace(-math.pi,math.pi, 500)
y = np.sin(x)

# RANDOMLY INITIALIZE WEIGHTS
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()
print(a,b,c,d); # exit()

learning_rate = 1e-6

#OPTIMIZATION LOOP
for t in range(15000):

    # FORWARD PASS: COMPUTE PREDICTED Y
    # MODEL: y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # COMPUTE AND PRINT LOSS
    loss = np.square(y_pred - y).sum()

    #REPORT
    if t % 100 == 0: print(t, loss)

    # BACKPROP TO COMPUTE GRADIENTS OF A, B, C, D WITH RESPECT TO LOSS
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.plot(x, y_pred, 'bo', label='True data', alpha=0.5)

plt.legend(loc='best')
plt.show()