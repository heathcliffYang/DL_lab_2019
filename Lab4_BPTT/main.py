from binary_addition import ba
import numpy as np
import rnn



alpha = 0.1
acc = 0
Epochs = 20000

rnn = rnn.RNN()

for i in range(100000000):
	train_data, label = ba()
	rnn.forward(train_data)
	rnn.backward(label, alpha)
	error = rnn.error()
	if error == 0:
		acc += 1
	if i%1000 == 0:
		print(i,error)
		acc /= 1000.
		print("Epoch ", i, " - accuracy is ",acc)
		acc = 0


