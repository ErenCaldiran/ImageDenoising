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
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import random_split

from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py

plt.style.use('fivethirtyeight')


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 351
batch_size = 64
learning_rate = 1e-3

transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),     #image = (image - mean) / std    makes it between -0.42 and 2.82
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

train_dataset = MNIST(root='./data',download=True, transform=transform_train,)

valid_dataset = MNIST(root='./data', train=False,download=True, transform=transform_train,)

print(len(train_dataset))
print(len(valid_dataset))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)        #Data/batchsize
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)        #Data/bathsize

print(len(train_loader))
print(len(valid_loader))

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20,kernel_size= 5,stride= 1)
        self.bn1=nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2=nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4 * 4 * 50, 250)
        self.fc11 = nn.Linear(250,16)
        #decoding
        self.fc22 = nn.Linear(16,250)
        self.fc2 = nn.Linear(250, 4 * 4 * 50)
        self.conv3 = nn.ConvTranspose2d(in_channels=50,out_channels=20,kernel_size=5,
                                           stride=1,padding=1,)

        self.conv4 = nn.ConvTranspose2d(in_channels=20,out_channels=10,kernel_size=5,
                                        stride=2,padding=1,)
                                        
        self.conv5 = nn.ConvTranspose2d(in_channels=10,out_channels=1,kernel_size=5,
                                        stride=2,padding=1,output_padding=1)
      
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))    #The formula for convolutional layers : Width = [ (Width – KernelWidth + 2*Padding) / Stride] + 1.
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)              #Below is the formula for Maxpool(Uses floor as default) :Hout= [ (Hin  + 2∗padding – dilation × (kernel_size − 1) – 1)/stride]
        x = torch.flatten(x, 1)                                                                         #Wout= [ (Win  + 2∗padding – dilation × (kernel_size − 1) – 1)/stride]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc22(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 50, 4, 4)                #The formula for transpose convolutional layers: Wout = stride(Win - 1) + kernelsize – 2*padding
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))               #all relu is better in this network
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss().to(device)   #Crossentropy better for classification??
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate) #Weight decay used to prevent oscillations, 4e-3 is ideal.

def train():
    model.train().cuda()  # put  in train mode #.to(device) in laptop
    total_loss = torch.zeros(1).to(device)
    for img, _ in train_loader:  # next batch

        img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
        #gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=1.6)
        #gaussian_img = torch.from_numpy(gaussian_img).to(device)
        saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.70)
        saltpepper_img = torch.from_numpy(saltpepper_img).to(device)

        img_ndarr = (img.cpu()).numpy()

        optimizer.zero_grad()

        output = model(saltpepper_img.float()).to(device)  # feed forward
        loss = criterion(output, img)    # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()
        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr,data_range=52)

        loss.backward()  # calculate new gradients
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate loss


    return saltpepper_img, img, output, total_loss, psnr


def valid():
    with torch.no_grad():
        model.eval().cuda()  #.to(device) in laptop
        valid_loss = torch.zeros(1).to(device)
        for img, _ in valid_loader:
            img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu

            #gaussian_image = skimage.util.random_noise(img.cpu(), mode="gaussian", var=1.6)
            #gaussian_image = torch.from_numpy(gaussian_image).to(device)
            saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.70)
            saltpepper_img = torch.from_numpy(saltpepper_img).to(device)
            # image, labels = image.to(device), labels.to(device)
            output = model(saltpepper_img.float().to(device))
            valid_loss += criterion(output, img)  # calculate loss
            img_ndarr = (img.cpu()).numpy()
            output_ndarr = (output.cpu().detach()).numpy()
            psnr = peak_signal_noise_ratio(img_ndarr, output_ndarr,data_range=52)


        return saltpepper_img, img, output, valid_loss, psnr


epocharray = []
trainlossarray = []
validlossarray = []
trainsnr = []
validsnr = []
inTotalData = 0

for epoch in range(num_epochs):
    noised_img, img, output, loss,psnr = train()
    valid_noised_img, valid_img, valid_output, valid_loss,valid_psnr = valid()

    epocharray.append(epoch)
    trainlossarray.append(loss.item() / len(train_loader))
    validlossarray.append(valid_loss.item() / len(valid_loader))
    trainsnr.append((psnr))
    validsnr.append((valid_psnr))

    print('epoch [{}/{}], loss:{:.4f}, SNR:{}'
          .format(epoch + 1, num_epochs, loss.item() / len(train_loader), psnr))
    print('Validation_loss:{}, SNR: {}'
          .format(valid_loss.item() / len(valid_loader), valid_psnr))
    if epoch % 50 == 0:
        pic_org = (img)
        pic_noised = (noised_img)
        pic_pred = (output)
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
                      name="Train loss",
                      marker={'color': 'blue', 'symbol': 100, 'size': 3},
                      mode="lines")

validErr = go.Scatter(x=epocharray,
                      y=validlossarray,
                      name="Valid loss",
                      marker={'color': 'red', 'symbol': 100, 'size': 3},
                      mode="lines")

inTotalData = [trainErr, validErr]

layout = dict(title='Train and validation loss',
              xaxis=dict(title='Epoch'),
              yaxis=dict(title='Loss'),
              )

InTotalfigure = dict(data=inTotalData, layout=layout)

py.plot(InTotalfigure, filename='secondtanhgauss.html', show_link=True)

trainSNR = go.Scatter(x=epocharray,
                      y=trainsnr,
                      name="Train snr",
                      marker={'color': 'blue', 'symbol': 100, 'size': 3},
                      mode="lines")

validSNR = go.Scatter(x=epocharray,
                      y=validsnr,
                      name="Valid snr",
                      marker={'color': 'red', 'symbol': 100, 'size': 3},
                      mode="lines")

TotalData = [trainSNR, validSNR]

SNRlayout = dict(title='Train and validation snr',
                 xaxis=dict(title='Epoch'),
                 yaxis=dict(title='Loss'),
                 )

Totalfigure = dict(data=TotalData, layout=SNRlayout)

py.plot(Totalfigure, filename='secondtanhgaussSNR.html', show_link=True)
