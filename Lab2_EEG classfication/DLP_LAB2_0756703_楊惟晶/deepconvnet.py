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
batch_size = 64
learning_rate = 1e-2
log_step = 100
plot_acc = np.zeros((6,epoch))
epoch_list = [x for x in range(1,epoch+1)]

class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        
        self.conv0 = nn.Conv2d(1, 25,
                               kernel_size=(1, 5), stride=(1, 1), bias=False)
        # First stack of layers
        self.conv1 = nn.Conv2d(25, 25,
                               kernel_size=(2, 1), stride=(1, 1), bias=False)
        self.batch1 = nn.BatchNorm2d(
            25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation == 0:
            self.elu1 = nn.ELU(alpha=1.0)
        elif activation == 1:
            self.elu1 = nn.ReLU()
        else:
            self.elu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2))
        self.drop1 = nn.Dropout(p=0.5)

        # Second stack of layers
        self.conv2 = nn.Conv2d(25, 50,
                               kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.batch2 = nn.BatchNorm2d(
            50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation == 0:
            self.elu2 = nn.ELU(alpha=1.0)
        elif activation == 1:
            self.elu2 = nn.ReLU()
        else:
            self.elu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool2d(kernel_size=(1,2))
        self.drop2 = nn.Dropout(p=0.5)

        # Third stack of layers
        self.conv3 = nn.Conv2d(50, 100,
                               kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.batch3 = nn.BatchNorm2d(
            100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation == 0:
            self.elu3 = nn.ELU(alpha=1.0)
        elif activation == 1:
            self.elu3 = nn.ReLU()
        else:
            self.elu3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2))
        self.drop3 = nn.Dropout(p=0.5)

        # 4th stack of layers
        self.conv4 = nn.Conv2d(100, 200,
                               kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.batch4 = nn.BatchNorm2d(
            200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if activation == 0:
            self.elu4 = nn.ELU(alpha=1.0)
        elif activation == 1:
            self.elu4 = nn.ReLU()
        else:
            self.elu4 = nn.LeakyReLU(negative_slope=0.01)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2))
        self.drop4 = nn.Dropout(p=0.5)
        
        # Classify
        self.linear = nn.Linear(in_features=200*43, out_features=2, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        #
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        #
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        #
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.elu4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


# input data
train_data, train_label, test_data, test_label = dl.read_bci_data()

# model initialization
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


for acti in range(3):
    model = DeepConvNet(activation=acti).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.3)
    # 1. 
    # Loss = nn.MSELoss()
    # 2. 
    Loss = nn.CrossEntropyLoss()

    epoch_acc = []

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
        print("Epoch", e, " - accuracy: ", plot_acc[acti*2][e], "%")

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
            torch.save(model.state_dict(), "deepconvnet_model/a_"+str(acti)+"/DeepConvNet_Epoch_"+str(e)+"_.pt")
            
            

        if plot_acc[acti*2+1][e] > 82:
            torch.save(model.state_dict(), "deepconvnet_model/a_"+str(acti)+"/DeepConvNet_Epoch_"+str(e)+"_"+str(100*acc/test_data.shape[0])+"_.pt")


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
plt.title("Activation function comparison (DeepConvNet)")
plt.savefig("DeepConvNet")

