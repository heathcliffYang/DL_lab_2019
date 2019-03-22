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


def derivative_relu(x):
    x_de_relu = np.zeros(x.shape)
    for i in range(x.shape[0]):
        if (x[0, i] <= 0):
            x_de_relu[0, i] = 0
        else:
            x_de_relu[0, i] = 1.
    return x_de_relu


class net():
    def __init__(self):
        # initialize netowrk weights
        self.w_1 = np.random.uniform(0.01, 2, (4, 2))
        self.w_2 = np.random.uniform(0.01, 2, (4, 4))
        self.w_3 = np.random.uniform(0.01, 2, (1, 4))

    def forward(self, x):

        self.x = x
        self.h_1 = np.matmul(self.w_1, x)

        self.h_1_p = sigmoid(self.h_1)

        self.h_2 = np.matmul(self.w_2, self.h_1_p)

        self.h_2_p = sigmoid(self.h_2)

        self.h_3 = np.matmul(self.w_3, self.h_2_p)

        output = sigmoid(self.h_3)
        # print("h1:", self.h_1, self.h_1.shape)
        # print("h2", self.h_2, self.h_2.shape)
        # print("h2'", self.h_2_p, self.h_2_p.shape)
        # print("h3", self.h_3, self.h_3.shape)
        # print("output", output)

        return output, self.h_3

    def loss(self, y_gt, z):
        return softplus((1-2*y_gt) * z)

    def peek_weights(self):
        print("Weights: \n", self.w_1, "\n", self.w_2, "\n", self.w_3, "\n")

    def back_propagation(self, y_gt, y_pred, z, learning_rate):
        # self.peek_weights()
               # derivative of softplus
        derivative_softplus = 1/(1+np.exp(-(1-2*y_gt)*z))
        g_w3 = derivative_softplus * \
            (1-2*y_gt) * np.transpose(self.h_2_p)
        self.w_3 = np.subtract(self.w_3, g_w3 * learning_rate)

        A = np.transpose(
            self.w_3 * derivative_sigmoid(np.transpose(self.h_2_p)))

        g_w2 = derivative_softplus * \
            (1-2*y_gt) * np.matmul(A, np.transpose(self.h_1_p))
        self.w_2 = np.subtract(self.w_2, g_w2 * learning_rate)

        g_w1 = derivative_softplus * (1-2*y_gt) * \
            np.matmul(
            np.transpose(
                np.matmul(
                    self.w_3 * derivative_sigmoid(np.transpose(self.h_2_p)), self.w_2)
                * derivative_sigmoid(np.transpose(self.h_1_p))), np.transpose(self.x))

        self.w_1 = np.subtract(self.w_1, g_w1 * learning_rate)
        # print("gradient:")
        # print(g_w1, g_w1.shape)
        # print(g_w2, g_w2.shape)
        # print(g_w3, g_w3.shape)
