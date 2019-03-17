import numpy as np
import data_generators as dg
import neural_netowrk


# prepare data
num_of_linear_dots = 5

x_1, y_1 = dg.generate_linear(n=num_of_linear_dots)
x_2, y_2 = dg.generate_XOR_easy()

arr = np.arange(num_of_linear_dots + 21)
np.random.shuffle(arr)

X = np.vstack((x_1, x_2))
Y = np.vstack((y_1, y_2))

X = X[arr]
Y = Y[arr]

Loss = 0

net = neural_netowrk.net()

# training
while True:
    for i in range(X.shape[0]):
        # forward
        y_pred = net.forward(X[i])
        # loss_function
        Loss = net.loss(Y[i], y_pred)
        # back-propagation
        print(y_pred, Loss)
    break
