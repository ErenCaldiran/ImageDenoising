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


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 200
batch_size = 128
learning_rate = 3e-4

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transform_train = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

#trainset= MNIST(root='./data', download=True, transform=transform_train)

# load the dataset
train_dataset = MNIST(
    root='./data', train=True,
    download=True, transform=transform_train,)

valid_dataset = MNIST(
    root='./data', train=True,
    download=True, transform=transform_train,)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))

#print(np.random.seed())   #these lines?
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

print(len(train_idx))
print(len(valid_idx))


#validation_set = torch.utils.data.random_split(trainset, 0.6)
#dataset = MNIST('./data', transform=img_transform, download=True)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler)        #Data/batchsize
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, sampler=valid_sampler)        #Data/bathsize

#dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

'''
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # ====== ENCODER PART ======
        # MNIST image is 1x28x28 (CxHxW)
        # Pytorch expects input data as BxCxHxW
        # B: Batch size
        # C: number of channels gray scale images have 1 channel
        # W: width of the image
        # H: height of the image

        # use 32 3x3 filters with padding
        # padding is set to 1 so that image W,H is not changed after convolution
        # stride is 2 so filters will move 2 pixels for next calculation
        # W after conv2d  [(W - Kernelw + 2*padding)/stride] + 1
        # after convolution we'll have Bx32 14x14 feature maps
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1)

        nn.Dropout(0.5)

        # after convolution we'll have Bx64 7x7 feature maps
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1
                               )

        nn.Dropout(0.5)

        # first fully connected layer from 64*7*7=3136 input features to 16 hidden units
        self.fc1 = nn.Linear(in_features=64 * 7 * 7,
                             out_features=16)

        nn.Dropout(0.5)

        self.fc2 = nn.Linear(in_features=16,
                             out_features=64 * 7 * 7)

        # 32 14x14
        self.conv_t1 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=32,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)

        # 1 28x28
        self.conv_t2 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=1,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten feature maps, Bx (CxHxW)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)  # reshape back to feature map format
        x = F.relu(self.conv_t1(x))
        x = torch.tanh(self.conv_t2(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=3e-4)

def train(model, loader, loss_func, optimizer):
    model.train().to(device)  # put  in train mode
    total_loss = torch.zeros(1).to(device)
    for img, _ in loader:  # next batch
        img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
        gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=2)
        gaussian_img = torch.from_numpy(gaussian_img).to(device)
        #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.45)
        #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)

        img_ndarr = (img.cpu()).numpy()

        #noise = torch.randn(*img.shape).to(device)  # generate random noise
        #noised_img = img.masked_fill(noise > 0.5, 1)  # set image values at indices where noise >0.5  to 1
        output = model(gaussian_img.float()).to(device)  # feed forward
        loss = loss_func(output, img)  # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()
        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr)

        loss.backward()  # calculate new gradients
        optimizer.step()  # update weights
        optimizer.zero_grad()  # clear previous gradients, backpropagation
        total_loss += loss  # accumulate loss
    return gaussian_img, img, output, total_loss, psnr


def valid(model):
    model.eval().to(device)  # Sets Train=False
    valid_loss = torch.zeros(1).to(device)
    with torch.no_grad():
        for img, _ in valid_loader:
            img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
            # image, labels = img
            # image = (img.cpu()).numpy()
            gaussian_image = skimage.util.random_noise(img.cpu(), mode="gaussian", var=2)
            gaussian_image = torch.from_numpy(gaussian_image).to(device)
            # image, labels = image.to(device), labels.to(device)
            output = model(gaussian_image.float().to(device))
            valid_loss += criterion(output, img)  # calculate loss
            img_ndarr = (img.cpu()).numpy()
            output_ndarr = (output.cpu().detach()).numpy()
            psnr = peak_signal_noise_ratio(img_ndarr, output_ndarr)
        return gaussian_image, img, output, valid_loss, psnr


'''
def valid(model,loader,loss_func):
    model.eval().to(device)
    total_loss = torch.zeros(1).to(device)
    for img, _ in loader:  # next batch
        img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
        gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=2)
        gaussian_img = torch.from_numpy(gaussian_img).to(device)
        img_ndarr = (img.cpu()).numpy()
        output = model(gaussian_img.float()).to(device)  # feed forward
        loss = loss_func(output, img)  # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()
        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr)
        loss.backward()  # calculate new gradients
        total_loss += loss  # accumulate loss

    return gaussian_img, img, output, total_loss, psnr
'''

for epoch in range(num_epochs):
    noised_img, img, output, loss, psnr = train(model, train_loader, criterion, optimizer)

   # valid_noised_img, valid_img, valid_output, valid_loss, valid_psnr = valid(model, valid_loader,criterion)
    # log
    print('epoch [{}/{}], loss:{:.4f}, SNR:{}'
          .format(epoch + 1, num_epochs, loss.item()/48000, psnr))
    if epoch % 10 == 0:
        pic_org = to_img(img.cpu().data)
        pic_noised = to_img(noised_img.cpu().data)
        pic_pred = to_img(output.cpu().data)
        save_image(pic_org, './denoise_image_org__{}.png'.format(epoch))
        save_image(pic_noised, './denoise_image_noised__{}.png'.format(epoch))
        save_image(pic_pred, './denoise_image_pred__{}.png'.format(epoch))

        #ValidationLoss function starts
    valid_noised_img, valid_img, valid_output, valid_loss, valid_psnr = valid(model)
    print('Validation_loss:{}, SNR: {}'
            .format(valid_loss.item()/12000, valid_psnr))
    if epoch % 10 == 0:
        valid_org = to_img(valid_img.cpu().data)
        valid_noisy = to_img(valid_noised_img.cpu().data)
        valid_pic = to_img(valid_output.cpu().data)
        save_image(valid_pic, './valid_denoise_image_pred{}.png'.format((epoch)))
        save_image(valid_noisy, './valid_denoise_image_noise_{}.png'.format((epoch)))
        save_image(valid_org, './valid_denoise_image_org_{}.png'.format((epoch)))

'''
    print('valid_epoch [{}/{}], valid_loss:{:.4f}, valid_SNR:{}'
          .format(epoch + 1, num_epochs, valid_loss.item()/60000, valid_psnr))
    if epoch % 10 == 0:
        pic_org = to_img(valid_img.cpu().data)
        pic_noised = to_img(valid_noised_img.cpu().data)
        pic_pred = to_img(valid_output.cpu().data)
        save_image(pic_org, './valid_denoise_image_org__{}.png'.format(epoch))
        save_image(pic_noised, './valid_denoise_image_noised__{}.png'.format(epoch))
        save_image(pic_pred, './valid_denoise_image_pred__{}.png'.format(epoch))
'''
# save the 
torch.save(model.state_dict(), './conv_autoencoder.pth')