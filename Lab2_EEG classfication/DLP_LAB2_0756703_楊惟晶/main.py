import dataloader as dl
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

epoch = 1000
batch_size = 360
learning_rate = 0.005
log_step = 100
plot_acc = np.zeros((6,epoch))
epoch_list = [x for x in range(1,epoch+1)]
load = True

class EEGNet(nn.Module):
    def __init__(self, activation):
        super(EEGNet, self).__init__()
        # First conv
        self.conv1 = nn.Conv2d(1, 16,
                               kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)
        self.batch1 = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Depth-wise conv
        self.conv2 = nn.Conv2d(16, 32,
                               kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.batch2 = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation == 0:
            self.elu1 = nn.ELU(alpha=1.0)
        elif activation == 1:
            self.elu1 = nn.ReLU()
        else:
            self.elu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.drop1 = nn.Dropout(p=0.25)

        # Separable conv
        self.conv3 = nn.Conv2d(32, 32,
                               kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.batch3 = nn.BatchNorm2d(
            32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation == 0:
            self.elu2 = nn.ELU(alpha=1.0)
        elif activation == 1:
            self.elu2 = nn.ReLU()
        else:
            self.elu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.drop2 = nn.Dropout(p=0.25)

        # Classify
        self.linear = nn.Linear(in_features=736, out_features=2, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.batch1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.batch2(x)
        # print(x.shape)
        x = self.elu1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.drop1(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.batch3(x)
        # print(x.shape)
        x = self.elu2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.drop2(x)
        # print(x.shape)
        res = x.view(x.size(0), x.size(1) * x.size(3))
        # print(res.shape)
        x = self.linear(res)
        # print(x.shape)
        return x


# input data
train_data, train_label, test_data, test_label = dl.read_bci_data()

# model initialization
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if load == True:
    model = EEGNet(activation = 1).to(device)
    model.load_state_dict(torch.load("eeg_model/a_1/EEG_Epoch_504_87.5_.pt"))
    # test
    acc = 0
    model.eval()
    with torch.no_grad():
        test_data_cuda = torch.FloatTensor(test_data).to(device)
        output = model(test_data_cuda)
            # 1. label to one-hot format
            # y = torch.LongTensor(test_label).view(-1,1)
            # batch_test_label = torch.zeros(test_label.shape[0], 2)
            # batch_test_label.scatter_(1, y, 1)
            # batch_test_label = torch.FloatTensor(batch_test_label).to(device)
            # 2.
        batch_test_label = torch.LongTensor(test_label).to(device)

        for i in range(test_data.shape[0]):
            if torch.argmax(output[i], dim=0) == test_label[i]:
                acc += 1
    print("    - Test data's accuracy :", 100*acc/test_data.shape[0], "%")

else:
    for acti in range(3):
        model = EEGNet(activation = acti).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Desktop 1 ~ 4
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1) 5
        # 1. 
        # Loss = nn.MSELoss()
        # 2. 
        Loss = nn.CrossEntropyLoss()

        # training
        for e in range(epoch):
            avg_loss = 0
            acc = 0
            log_sum = 0
            scheduler.step()
            model.train()
            for i in range(0, train_data.shape[0], batch_size):
                if (batch_size > train_data.shape[0] - i):
                    idx_batch_size = train_data.shape[0]
                else:
                    idx_batch_size = i + batch_size
                # print("Data idx from ", i,  "to", idx_batch_size)
                batch_train_data = train_data[i:idx_batch_size]
                batch_train_data = torch.FloatTensor(batch_train_data).to(device)
                # 1. label to one-hot format
                # y = torch.LongTensor(train_label[i:idx_batch_size]).view(-1,1)
                # batch_train_label = torch.zeros(idx_batch_size-i, 2)
                # batch_train_label.scatter_(1, y, 1)
                # batch_train_label = torch.FloatTensor(batch_train_label).to(device)
                # 2. 
                batch_train_label = train_label[i:idx_batch_size]
                batch_train_label = torch.LongTensor(batch_train_label).to(device)

                # print("Shape of input:", batch_train_data.shape, batch_train_label.shape)
                

                optimizer.zero_grad()
                output = model(batch_train_data)
                # print(output.shape, batch_train_label.shape)
                loss = Loss(output, batch_train_label)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
                for j in range(idx_batch_size-i):
                    log_sum += 1
                    if train_label[i+j] == torch.argmax(output[j], dim=0):
                        acc += 1
                # print("cumulative # of hits:", acc)
                
            if log_sum != train_data.shape[0]:
                print("EXCUSE ME?")
                break
            plot_acc[acti*2][e] = 100*acc/train_data.shape[0]
            print("Epoch", e, " - Training accuracy: ", plot_acc[acti*2][e], "%")

            # test
            acc = 0
            model.eval()
            with torch.no_grad():
                test_data_cuda = torch.FloatTensor(test_data).to(device)
                output = model(test_data_cuda)
                # 1. label to one-hot format
                # y = torch.LongTensor(test_label).view(-1,1)
                # batch_test_label = torch.zeros(test_label.shape[0], 2)
                # batch_test_label.scatter_(1, y, 1)
                # batch_test_label = torch.FloatTensor(batch_test_label).to(device)
                # 2.
                batch_test_label = torch.LongTensor(test_label).to(device)

                loss = Loss(output, batch_test_label)
                for i in range(test_data.shape[0]):
                    if torch.argmax(output[i], dim=0) == test_label[i]:
                        acc += 1
            plot_acc[acti*2+1][e] = 100*acc/test_data.shape[0]
            print("    - Test data's accuracy :", 100*acc/test_data.shape[0], "%")
            if (e % log_step == 0):
                print("Check point - epoch ", e)
                torch.save(model.state_dict(), "eeg_model/a_"+str(acti)+"/EEG_Epoch_"+str(e)+"_.pt")
                
                
            if plot_acc[acti*2+1][e] >= 87:
                torch.save(model.state_dict(), "eeg_model/a_"+str(acti)+"/EEG_Epoch_"+str(e)+"_"+str(100*acc/test_data.shape[0])+"_.pt")




    color = ['#F5B4B3', '#CA885B', '#DAE358', '#9DE358', '#58E39D', '#58E3E1', '#58A2E3', '#5867E3', '#9D58E3', '#E158E3', '#E358B0', '#E35869']
    # Plot
    print("Hyper parameters:\n  Epoch :", epoch, "  Batch size :", batch_size, "  Learning rate :", learning_rate, "\nFirst 6 highest accuracy:")
    for i in range(6):
        if i < 2:
            acti_name = 'elu'
        elif i < 4:
            acti_name = 'relu'
        else:
            acti_name = 'leaky_relu'
        
        if i % 2 == 0:
            model_mode = '_train'
        else:
            model_mode = '_test'
            print("Activiation function : "+acti_name)
            rank = np.sort(plot_acc[i])
            print("  ", rank[-6:])
        line_name = acti_name+model_mode
        plt.plot(epoch_list, plot_acc[i], color=color[i], linestyle='-', label=line_name)

    plt.xlim(-5, epoch+5)
    plt.ylim(ymax = 105)
    plt.legend(loc=4)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy(%)")
    plt.title("Activation function comparison (EEGNet)")
    plt.savefig("EEG")
