import os
import skimage
from skimage.metrics import peak_signal_noise_ratio
import torch
import torchvision
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F

from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py

import numpy as np
import csv

plt.style.use('fivethirtyeight')

from torch.utils.data.sampler import SubsetRandomSampler


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 351
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transform_train = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))   #image = (image - mean) / std    makes it between -1 and 1
])

#trainset= MNIST(root='./data', download=True, transform=transform_train)

train_dataset = MNIST(root='./data',download=True, transform=transform_train)

valid_dataset = MNIST(root='./data', train=False,download=True, transform=transform_train)

print(len(train_dataset))
print(len(valid_dataset))

#test_ds, valid_ds = torch.utils.data.random_split(train_dataset, (50000, 10000))

#print(len(test_ds))
#print(len(valid_ds))


#validation_set = torch.utils.data.random_split(trainset, 0.6)
#dataset = MNIST('./data', transform=img_transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)        #Data/batchsize
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)        #Data/bathsize


print(len(train_loader))
print(len(valid_loader))
#dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,
                               kernel_size=3,stride=2,padding=1)
        self.bn1=nn.BatchNorm2d(32)
        # after convolution we'll have Bx64 7x7 feature maps
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,
                               kernel_size=3,stride=2,padding=1
                               )
        self.bn2=nn.BatchNorm2d(64)
        # first fully connected layer from 64*7*7=3136 input features to 16 hidden units
        self.fc11 = nn.Linear(in_features=64*7*7,out_features=196)

        self.fc1 = nn.Linear(in_features=196,out_features=16)

        self.fc2 = nn.Linear(in_features=16,out_features=196)

        self.fc22 = nn.Linear(in_features=196,out_features=64*7*7)
        # 32 14x14
        self.conv_t1 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,
                                          stride=2,padding=1,output_padding=1)
        # 1 28x28
        self.conv_t2 = nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=3,
                                          stride=2,padding=1,output_padding=1)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten feature maps, Bx (CxHxW)
        x = torch.tanh(self.fc11(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc22(x))
        x = x.view(-1, 64, 7, 7)  # reshape back to feature map format
        x = torch.tanh(self.conv_t1(x))
        x = torch.tanh(self.conv_t2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate) #Weight decay used to prevent oscillations, 4e-3 is ideal.

def train():
    model.train().cuda()  # put  in train mode #.to(device) in laptop
    total_loss = torch.zeros(1).to(device)
    for img, _ in train_loader:  # next batch
        img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
        gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=1.6)
        gaussian_img = torch.from_numpy(gaussian_img).to(device)
        #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.45)
        #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)

        img_ndarr = (img.cpu()).numpy()

        optimizer.zero_grad()

        output = model(gaussian_img.float()).to(device)  # feed forward
        loss = criterion(output, img)    # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()

        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr,data_range=14)

        loss.backward()  # calculate new gradients
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate loss

    return gaussian_img, img, output, total_loss, psnr


def valid():
    with torch.no_grad():
        model.eval().cuda()  #.to(device)
        valid_loss = torch.zeros(1).to(device)
        for img, _ in valid_loader:
            img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu

            gaussian_image = skimage.util.random_noise(img.cpu(), mode="gaussian", var=1.6)
            gaussian_image = torch.from_numpy(gaussian_image).to(device)
            #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.45)
            #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)
            # image, labels = image.to(device), labels.to(device)
            output = model(gaussian_image.float().to(device))
            valid_loss += criterion(output, img)  # calculate loss
            img_ndarr = (img.cpu()).numpy()
            output_ndarr = (output.cpu().detach()).numpy()
            psnr = peak_signal_noise_ratio(img_ndarr, output_ndarr,data_range=14)

        return gaussian_image, img, output, valid_loss, psnr


epocharray = []
trainlossarray = []
validlossarray = []
trainsnr = []
validsnr = []
inTotalData = 0

for epoch in range(num_epochs):
    noised_img, img, output, loss, psnr = train()
    valid_noised_img, valid_img, valid_output, valid_loss, valid_psnr = valid()

    epocharray.append(epoch)
    trainlossarray.append(loss.item()/len(train_loader))
    validlossarray.append(valid_loss.item()/len(valid_loader))
    trainsnr.append((psnr))
    validsnr.append((valid_psnr))

    print('epoch [{}/{}], loss:{:.4f}, SNR:{}'
        .format(epoch + 1, num_epochs, loss.item()/len(train_loader), psnr))
    print('Validation_loss:{}, SNR: {}'
        .format(valid_loss.item() / len(valid_loader), valid_psnr))
    if epoch % 50 == 0:
        pic_org = (img)
        pic_noised = (noised_img)
        pic_pred = (output)

        '''
        img = (img.cpu().detach()).numpy()
        noised_img = (noised_img.cpu().detach()).numpy()
        output = (output.cpu().detach()).numpy()

        pixval1 = list(img)     #0 and 1
        pixval2 = list(noised_img)  #0 and 1
        pixval3 = (output).tolist()  #-0.5 and 1

        print(img.shape)
        print(noised_img.shape)
        print(output.shape)

        
        with open("try1.txt", "w") as txt_file:
            for line in str(pixval1):
                txt_file.write(" ".join(line))
        with open("try2.txt", "w") as txt_file:
            for line in str(pixval2):
                txt_file.write(" ".join(line))
        with open("try3.txt", "w") as txt_file:
            for line in str(pixval3):
             txt_file.write(" ".join(line))

        rows = []
        with open('try3.txt', mode='r') as infile:
            reader = csv.reader(infile, delimiter=",")
            for row in reader:  # each row is a list
                rows.append(row)

        # lambda function to filter min considering the second column
        minimus = min(rows, key=lambda x: (x[:]))

        # done
        print(minimus)
'''
        save_image(pic_org, './denoise_image_org__{}.png'.format(epoch))
        save_image(pic_noised, './denoise_image_noised__{}.png'.format(epoch))
        save_image(pic_pred, './denoise_image_pred__{}.png'.format(epoch))
        valid_org = (valid_img)
        valid_noisy = (valid_noised_img)
        valid_pic = (valid_output.cpu().data)
        save_image(valid_pic, './valid_denoise_image_pred{}.png'.format((epoch)))
        save_image(valid_noisy, './valid_denoise_image_noise_{}.png'.format((epoch)))
        save_image(valid_org, './valid_denoise_image_org_{}.png'.format((epoch)))

trainErr = go.Scatter(x=epocharray,
                            y=trainlossarray,
                            name = "Train loss",
                            marker={'color': 'blue', 'symbol': 100, 'size': 3},
                            mode="lines")

validErr = go.Scatter(x=epocharray,
                            y=validlossarray,
                            name = "Valid loss",
                            marker={'color': 'red', 'symbol': 100, 'size': 3},
                            mode="lines")

inTotalData = [trainErr,validErr]

layout = dict(title = 'Train and validation loss',
              xaxis = dict(title = 'Epoch'),
              yaxis = dict(title = 'Loss'),
              )

InTotalfigure = dict(data=inTotalData, layout=layout)

py.plot(InTotalfigure, filename='SaltandPepperboth045.html', show_link=True)


trainSNR = go.Scatter(x=epocharray,
                           y=trainsnr,
                           name = "Train snr",
                           marker={'color': 'blue', 'symbol': 100, 'size': 3},
                           mode="lines")

validSNR = go.Scatter(x=epocharray,
                      y=validsnr,
                      name="Valid snr",
                      marker={'color': 'red', 'symbol': 100, 'size': 3},
                      mode="lines")

TotalData = [trainSNR,validSNR]

SNRlayout = dict(title = 'Train and validation snr',
              xaxis = dict(title = 'Epoch'),
              yaxis = dict(title = 'Loss'),
              )

Totalfigure = dict(data=TotalData, layout=SNRlayout)

py.plot(Totalfigure, filename='SNRboth045.html', show_link=True)

torch.save(model.state_dict(), './conv_autoencoder.pth')