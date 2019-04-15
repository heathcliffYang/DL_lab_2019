from __future__ import print_function
from argparse import ArgumentParser
import dataloader as dl
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils import data
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import net
import confusion_matrix_tool as cmt


# Check CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("CUDA Device:", device)

# 1. - 1 Hyper parameters parser for single set of parameters
parser = ArgumentParser()
parser.add_argument("epoch", help="resnet18 needs 10 epochs and resnet50 needs 5", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("lr", type=float)
parser.add_argument("momentum", type=float)
parser.add_argument("weight_decay", type=float)

args = parser.parse_args()

# Some needed by plot the results
plot_acc = np.zeros((8, args.epoch))
epoch_list = [x for x in range(1,args.epoch+1)]
acc = 0
log_step = 5

# 2. load model
models = []
models.append(net.resnet18(True, 5).to(device))
models.append(net.resnet18(False, 5).to(device))
models.append(net.resnet50(True, 5).to(device))
models.append(net.resnet50(False, 5).to(device))
print(models[0])
print(models[2])
optimizers = []
for i in range(4):
    optimizers.append(optim.SGD(models[i].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay))
Loss = nn.CrossEntropyLoss()

# 3. training
train_data = dl.RetinopathyLoader(root='data/', mode='train')
if len(train_data) != 28099:
    print("Training data is not intact.")
    raise EOFError
test_data = dl.RetinopathyLoader(root='data/', mode='test')
if len(test_data) != 7025:
    print("Testing data is not intact.")
    raise EOFError

for m in range(4):
    for e in range(args.epoch):
        acc = 0
        log_sum = 0
        models[m].train()
        for i in range(0, len(train_data), args.batch_size):
            img_batch = []
            label_batch =[]
            if (args.batch_size > len(train_data) - i):
                idx_batch_size = len(train_data)
            else:
                idx_batch_size = i+args.batch_size

            for j in range(i, idx_batch_size):
                img, label = train_data[j]
                img_batch.append(img)
                label_batch.append(label)
            
            img_batch = torch.FloatTensor(np.asarray(img_batch)).to(device)
            label_batch = torch.LongTensor(np.asarray(label_batch)).to(device)

            # forward backward
            optimizers[m].zero_grad()
            output = models[m](img_batch)
            loss = Loss(output, label_batch)
            loss.backward()
            optimizers[m].step()
            if (i%40 == 0):
                print("From ", i, "to", idx_batch_size, "output", output.shape)
            # compute accuracy
            for j in range(idx_batch_size-i):
                log_sum += 1
                if label_batch[j] == torch.argmax(output[j], dim=0):
                    acc += 1
        if log_sum != len(train_data):
            print("Mismatch between # of data considered by accuracy computation and # of total training data")
            raise EOFError
        plot_acc[2*m][e] = 100*acc/len(train_data)
        print("Epoch", e, " - Training accuracy: ", plot_acc[2*m][e], "%")

        # 4. testing every epoch
        acc = 0
        models[m].eval()
        with torch.no_grad():
            for i in range(0, len(test_data), args.batch_size):
                img_batch = []
                label_batch =[]
                if (args.batch_size > len(test_data) - i):
                    idx_batch_size = len(test_data)
                else:
                    idx_batch_size = i+args.batch_size

                for j in range(i, idx_batch_size):
                    img, label = test_data[j]
                    img_batch.append(img)
                    label_batch.append(label)

                img_batch = torch.FloatTensor(np.asarray(img_batch)).to(device)
                label_batch = np.asarray(label_batch)
                if (i%40 == 0):
                    print("From ", i, "to", idx_batch_size)
                output = models[m](img_batch)
                if i == 0:
                    y_true = label_batch
                    y_pred = torch.argmax(output, dim=1).cpu().numpy()
                else:
                    y_true = np.concatenate((y_true, label_batch))
                    y_pred_batch = torch.argmax(output, dim=1).cpu().numpy()
                    y_pred = np.concatenate((y_pred, y_pred_batch))
                for j in range(idx_batch_size-i):
                    if torch.argmax(output[j], dim=0) == label_batch[j]:
                        acc += 1
            plot_acc[2*m+1][e] = 100*acc/len(test_data)
            print("    - Test data's accuracy :", plot_acc[2*m+1][e], "%")
            if (e % log_step == 0):
                print("Check point - epoch ", e)
                torch.save(models[m].state_dict(), "M_{}_e_{}_bs_{}_lr_{}_mom_{}_wd_{}_.pt".format(m, e, args.batch_size, args.lr, args.momentum, args.weight_decay))
                    
            if plot_acc[m*2+1][e] >= 82:
                torch.save(models[m].state_dict(), "M_{}_e_{}_bs_{}_lr_{}_mom_{}_wd_{}_acc_{}_.pt".format(m, e, args.batch_size, args.lr, args.momentum, args.weight_decay, plot_acc[m*2+1][e]))

            # confusion matrix
            print("y_pred's shape in confusion matrix: ", y_pred.shape, y_true.shape)
            cmt.plot_confusion_matrix(y_true, y_pred, classes=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'], normalize=True,title='Normalized confusion matrix')
            plt.savefig("Conf_matrix_M_{}_e_{}_bs_{}_lr_{}_mom_{}_wd_{}.png".format(m, e, args.batch_size, args.lr, args.momentum, args.weight_decay))



color = ['#F5B4B3', '#CA885B', '#DAE358', '#9DE358', '#58E39D', '#58E3E1', '#58A2E3', '#5867E3', '#9D58E3', '#E158E3', '#E358B0', '#E35869']
# Plot
print("Hyper parameters:\n", "Epoch_{}_bs_{}_lr_{}_mom_{}_wd_{}_.pt".format(e, args.batch_size, args.lr, args.momentum, args.weight_decay), "\nFirst 6 highest accuracy:")

for i in range(8):
    if i < 2:
        layer_num = '18_pre'
    elif i < 4:
        layer_num = '18'
    elif i < 6:
        layer_num = '50_pre'
    else:
        layer_num = '50'
        
    if i % 2 == 0:
        model_mode = '_train'
    else:
        model_mode = '_test'
        print("Resnet{}'s highest accuracy : {}".format(layer_num, np.max(plot_acc[i])))
    line_name = layer_num+model_mode
    plt.plot(epoch_list, plot_acc[i], color=color[i], linestyle='-', label=line_name)

plt.xlim(-5, args.epoch+5)
plt.ylim(ymax = 105)
plt.legend(loc=4)
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.title("Result Comparison")
plt.savefig("Result_Bs_{}_lr_{}_mom_{}_wd_{}".format(args.batch_size, args.lr, args.momentum, args.weight_decay))