import numpy as np
import math
import data_generators as dg
import neural_netowrk
import matplotlib.pyplot as plt


# prepare data
# num_of_linear_dots = 100
# X, Y = dg.generate_linear(n=num_of_linear_dots)
# x_1, y_1 = dg.generate_linear(n=num_of_linear_dots)

X, Y = dg.generate_XOR_easy()
x_1, y_1 = dg.generate_XOR_easy()

e = []
l = []

color = ['r-', 'b-', 'y-', 'g-', 'c-', 'k-', 'm-']
nodes = [2, 3, 4, 5, 8, 10, 20]
colors = ['ro', 'bo', 'yo', 'go', 'co', 'ko', 'mo']

for k in range(7):
    # neural network parameters
    Loss_avg = 0
    lr = 0.1
    acc = 0
    epoch = 0

    loss_pool = []

    net = neural_netowrk.net(nodes[k])

    # training
    while True:
        Loss_avg = 0
        acc = 0
        for i in [0, 1, 7, 8, 15, 16]:

            # for i in range(1):

            # forward
            y_pred, z = net.forward(np.expand_dims(X[i], axis=1))
            # loss_function
            Loss = net.loss(Y[i], z)
            Loss_avg += Loss[0, 0]

            if (y_pred > 0.5 and Y[i] > 0.5):
                acc += 1
            elif (y_pred < 0.5 and Y[i] < 0.5):
                acc += 1

            # back-propagation
            if (Loss[0, 0] > 0.1):
                net.back_propagation(Y[i], y_pred, z, learning_rate=lr)
        epoch += 1
        # Loss_avg /= float(X.shape[0])
        Loss_avg /= 6
        loss_pool.append(Loss_avg)
        # acc /= float(X.shape[0])
        acc /= 6

        if (epoch % 100 == 0):
            e.append(epoch)
            l.append(Loss_avg)
            print("Epoch ", epoch, " - loss : ", Loss_avg, " - ACC : ", acc)
            if (math.fabs(sum(loss_pool)/float(len(loss_pool)) - Loss_avg) < 0.001 and Loss_avg < 0.1):
                net.peek_weights()
                print("End")
                break
            elif (Loss_avg < 1.):
                lr = 0.05
            elif (Loss_avg < 0.5):
                lr = 0.01
            elif (Loss_avg < 0.3):
                lr = 0.0001
            elif (math.isnan(Loss_avg)):
                print("Oh no!")
                break
            loss_pool = []

            if (epoch > 20000):
                break
                # Log per 1000 epochs

        if (epoch % 1000 == 0):
            net.peek_weights()

    plt.subplot(1, 2, 1)
    plt.title('Training', fontsize=18)
    plt.plot(e, l, color[k], label=str(nodes[k])+'_nodes')

    e = []
    l = []

    # testing
    Y_pred = []
    acc = 0
    for i in range(x_1.shape[0]):
        y_pred, z = net.forward(np.expand_dims(x_1[i], axis=1))
        Y_pred.append(y_pred)
        if (y_pred > 0.5 and y_1[i] > 0.5):
            print(i, " - True - Y_pred : ", y_pred, "Y_gt : ", y_1[i])
            acc += 1
        elif (y_pred < 0.5 and y_1[i] < 0.5):
            print(i, " - True - Y_pred : ", y_pred, "Y_gt : ", y_1[i])
            acc += 1
    print("ACC : ", acc/x_1.shape[0])
    plt.subplot(1, 2, 2)
    plt.title('Test', fontsize=18)
    plt.plot(k, acc/x_1.shape[0], colors[k], label=str(nodes[k])+'_nodes')
    plt.ylim(0, 1.5)


plt.legend()
plt.savefig("nodes")
