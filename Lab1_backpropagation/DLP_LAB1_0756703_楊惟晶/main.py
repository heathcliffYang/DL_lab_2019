import numpy as np
import math
import data_generators as dg
import neural_netowrk


# prepare data
num_of_linear_dots = 100

x_1, y_1 = dg.generate_linear(n=num_of_linear_dots)
x_2, y_2 = dg.generate_XOR_easy()

arr = np.arange(num_of_linear_dots + 21)
np.random.shuffle(arr)

X = np.vstack((x_1, x_2))
Y = np.vstack((y_1, y_2))

X = X[arr]
Y = Y[arr]


# neural network parameters
Loss_avg = 0
lr = 1
acc = 0
epoch = 0

loss_pool = []

net = neural_netowrk.net()

# training
while True:
    Loss_avg = 0
    acc = 0
    for i in range(X.shape[0]):

        # for i in range(1):

        # forward
        y_pred, z = net.forward(np.expand_dims(X[i], axis=1))
        # loss_function
        Loss = net.loss(Y[i], z)
        Loss_avg += Loss[0, 0]
        # back-propagation
        if (Loss[0, 0] > 0.1):
            # net.peek_weights()
            net.back_propagation(Y[i], y_pred, z, learning_rate=lr)
            # net.peek_weights()
            # print("i : ", i, "data : ", X[i], "Loss : ", Loss,
            #       "Y_pred : ", y_pred, "Y_gt : ", Y[i], "\n")
    epoch += 1
    Loss_avg /= float(X.shape[0])
    loss_pool.append(Loss_avg)

    if (epoch % 100 == 0):
        print("Epoch ", epoch, " - loss : ", Loss_avg)
        if (math.fabs(sum(loss_pool)/float(len(loss_pool)) - Loss_avg) < 0.001):
            print("End")
            break
        elif (Loss_avg < 1.):
            lr = 0.7
        elif (Loss_avg < 0.5):
            lr = 0.1
        elif (Loss_avg < 0.3):
            lr = 0.01
        elif (math.isnan(Loss_avg)):
            print("Oh no!")
            break
        # Log per 1000 epochs
        loss_pool = []

    if (epoch % 1000 == 0):
        net.peek_weights()


# testing
x_1, y_1 = dg.generate_linear(n=num_of_linear_dots)
Y_pred = []
for i in range(x_1.shape[0]):
    y_pred, z = net.forward(np.expand_dims(x_1[i], axis=1))
    Y_pred.append(y_pred)
    print("i : ", i, "- data : ", x_1[i],
          "Y_pred : ", y_pred, "Y_gt : ", y_1[i])

dg.show_result(x_1, y_1, Y_pred, "linear")

x_2, y_2 = dg.generate_XOR_easy()
Y_pred = []
for i in range(x_2.shape[0]):
    y_pred, z = net.forward(np.expand_dims(x_2[i], axis=1))
    Y_pred.append(y_pred)
    print("i : ", i, "- data : ", x_2[i],
          "Y_pred : ", y_pred, "Y_gt : ", y_2[i])

dg.show_result(x_2, y_2, Y_pred, "XOR")
