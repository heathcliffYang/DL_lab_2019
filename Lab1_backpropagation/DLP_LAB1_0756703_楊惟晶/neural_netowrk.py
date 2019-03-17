import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0-x)


def softplus(x):
    return np.log(1+np.exp(x))


def relu(x):
    x_relu = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_relu[i] = max(0, x[i])
    return x_relu


class net():
    def __init__(self):
        # initialize netowrk weights
        self.w_1 = np.random.uniform(-1, 1, (2, 2))
        self.w_2 = np.random.uniform(-1, 1, (2, 2))
        self.w_3 = np.random.uniform(-1, 1, (1, 2))

    def forward(self, x):
        h_1 = np.matmul(self.w_1, x)
        h_1 = relu(h_1)
        h_2 = np.matmul(self.w_2, h_1)
        h_2 = relu(h_2)
        h_3 = np.matmul(self.w_3, h_2)
        output = sigmoid(h_3)

        return output

    def loss(self, y_gt, y_pred):
        return softplus((1-2*y_gt) * y_pred)

    def back_propagation(self, loss):
        g_w3 = derivative_sigmoid(h_3)
