import os
import skimage
from skimage.metrics import peak_signal_noise_ratio
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transform_train = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

trainset= MNIST(root='./data', download=True, transform=transform_train)

#dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


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

        # after convolution we'll have Bx64 7x7 feature maps
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1
                               )

        # first fully connected layer from 64*7*7=3136 input features to 16 hidden units
        self.fc1 = nn.Linear(in_features=64 * 7 * 7,
                             out_features=16)

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
net = AutoEncoder().to(device)
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
model = net.to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

def train(net, loader, loss_func, optimizer):
    net.train().to(device)  # put model in train mode
    total_loss = torch.zeros(1).to(device)
    for img, _ in loader:  # next batch
        img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
        gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=2)
        gaussian_img = torch.from_numpy(gaussian_img).to(device)
        saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.45)
        saltpepper_img = torch.from_numpy(saltpepper_img).to(device)

        img_ndarr = (img.cpu()).numpy()

        #noise = torch.randn(*img.shape).to(device)  # generate random noise
        #noised_img = img.masked_fill(noise > 0.5, 1)  # set image values at indices where noise >0.5  to 1
        output = net(gaussian_img.float()).to(device)  # feed forward
        loss = loss_func(output, img)  # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()
        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr)

        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # calculate new gradients
        optimizer.step()  # update weights
        total_loss += loss  # accumulate loss
    return gaussian_img, img, output, total_loss, psnr

for epoch in range(num_epochs):
    noised_img, img, output, loss, psnr = train(net, dataloader, criterion, optimizer)
    # log
    print('epoch [{}/{}], loss:{:.4f}, SNR:{}'
          .format(epoch + 1, num_epochs, loss.item()/60000, psnr))
    if epoch % 10 == 0:
        pic_org = to_img(img.cpu().data)
        pic_noised = to_img(noised_img.cpu().data)
        pic_pred = to_img(output.cpu().data)
        save_image(pic_org, './denoise_image_org__{}.png'.format(epoch))
        save_image(pic_noised, './denoise_image_noised__{}.png'.format(epoch))
        save_image(pic_pred, './denoise_image_pred__{}.png'.format(epoch))

# save the model
torch.save(net.state_dict(), './conv_autoencoder.pth')