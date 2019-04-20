import numpy as np
import math

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# def derivative_sigmoid(x):
#     return np.multiply(x, 1.0-x)

class RNN():
    def __init__(self):
        self.num_layers = 8
        self.dim_hidden = 16
        self.dim_output = 1
        self.init_weights()
        self.h = np.zeros((8+1, self.dim_hidden, 1))
        self.o = np.zeros((8, self.dim_output))
        self.y_hat = np.zeros((8, self.dim_output))
        self.gradient_h = np.zeros((8+1, self.dim_hidden, 1))
        self.gradient_W = np.zeros((self.dim_hidden,self.dim_hidden))
        self.gradient_U = np.zeros((self.dim_hidden, 2))
        self.gradient_V = np.zeros((self.dim_output, self.dim_hidden))

    def init_weights(self):
        dim_hidden = self.dim_hidden
        dim_output = self.dim_output

        self.U = np.random.uniform(-1, 1, (dim_hidden, 2))
        print("U", self.U)

        self.W = np.random.uniform(-1, 1, (dim_hidden,dim_hidden))
        print("W", self.W)

        self.V = np.random.uniform(-1, 1, (dim_output, dim_hidden))
        print("V", self.V)

    def forward(self, x):
        # x = [8 x 2 x 1] = [8 * [num1's digit i, num2's digit i]]
        self.x = x.reshape(8,2,1)
        # print("x", self.x)
        for i in range(len(x)):
            a = np.matmul(self.W, self.h[i]) + np.matmul(self.U, x[i].reshape(2,1))
            # print(i,"-a", a)
            self.h[i+1] = np.tanh(a)
            # print(i+1, "-h", self.h[i+1])
            self.o[i] = np.matmul(self.V, self.h[i+1])
            # print(i,"o", self.o[i])
            self.y_hat[i] = sigmoid(self.o[i])

        return self.y_hat

    def H(self, i):
        return np.diag(1-self.h[i,:,0]**2)

    def derivative_softplus(self, i):  
        if math.isnan(1/(1+np.exp(-(1-2*self.y_gt[i,0])*self.o[i]))):
            print("A", self.o[i])
        elif math.isnan(1 - 2*self.y_gt[i,0]):
            print("B")
            raise EOFError
        return 1/(1+np.exp(-(1-2*self.y_gt[i,0])*self.o[i])) * (1 - 2*self.y_gt[i,0]) 

    def error(self):
        error = 0
        for i in range(8):
            if (self.y_hat[i] > 0.5 and self.y_gt[i] < 0.5) or (self.y_hat[i] <= 0.5 and self.y_gt[i] > 0.5):
                error += 1
        return error
        

    def backward(self, y_gt, alpha):
        self.y_gt = y_gt
        self.gradient_W = np.zeros((self.dim_hidden,self.dim_hidden))
        self.gradient_U = np.zeros((self.dim_hidden, 2))
        self.gradient_V = np.zeros((self.dim_output, self.dim_hidden))
        for i in range(self.num_layers, 0, -1):
            # gradient L w.r.t 
            if i == self.num_layers:
                self.gradient_h[i] = self.V.T*self.derivative_softplus(i-1)
                # print(i,"-grad h L", self.gradient_h[i])
            else:
                self.gradient_h[i] = self.V.T*self.derivative_softplus(i-1) + np.matmul(np.matmul(self.W.T, self.H(i+1)), self.gradient_h[i+1])
                # print(i,"-grad h L", self.gradient_h[i])

            self.gradient_W = self.gradient_W + np.matmul(np.matmul(self.H(i), self.gradient_h[i]), self.h[i-1].T)
            self.gradient_U = self.gradient_U + np.matmul(np.matmul(self.H(i), self.gradient_h[i]), self.x[i-1].T)
            self.gradient_V = self.gradient_V + self.derivative_softplus(i-1)*self.h[i].T

        # print("grad W", self.gradient_W)
        # print("grad U", self.gradient_U)
        # print("grad V", self.gradient_V)

        self.W -= self.gradient_W * alpha
        self.U -= self.gradient_U * alpha
        self.V -= self.gradient_V * alpha







