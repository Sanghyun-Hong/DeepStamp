
"""
    Description: networks for the CIFAR10 watermark task
"""
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision
import torchvision.transforms as transforms


"""
    Watermark Network
    (eqn. G(x, w-noised) -> w')
"""
class Cifar10_Watermarker(nn.Module):
    def __init__(self, activation='PReLU', chdim=16):
        super(Cifar10_Watermarker, self).__init__()
        self.activation = activation
        self.convs = nn.Sequential(
            # 1st layer
            nn.Conv2d(7, chdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chdim),
            activation_layer(activation),
            # 2nd layer
            nn.Conv2d(chdim, chdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chdim),
            activation_layer(activation),
            # 3rd layer
            nn.Conv2d(chdim, chdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chdim),
            activation_layer(activation),
            # 4th layer
            nn.Conv2d(chdim, chdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chdim),
            activation_layer(activation),
            # 5th layer
            nn.Conv2d(chdim, 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.sigmoid(x)
        return x


"""
    Discriminator (D): the discriminator that catches real/fake watermarks
"""
class Cifar10_Discriminator(nn.Module):
    def __init__(self, chdim=4):
        super(Cifar10_Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 32 x 32
            nn.Conv2d(3, chdim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state chdim. (chdim*2) x 16 x 16
            nn.Conv2d(chdim, chdim * 2, 4, 2, 1),
            nn.BatchNorm2d(chdim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state chdim. (chdim*2) x 8 x 8
            nn.Conv2d(chdim * 2, 1, 8, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 1).squeeze(1)
        return output


"""
    Preserver (Vs): the auto-encoder that learns representation of input watermarks
    (Note: from the EB-GAN https://arxiv.org/pdf/1609.03126.pdf)
"""
class Cifar10_AutoEncoder(nn.Module):
    def __init__(self, chdim=8):
        super(Cifar10_AutoEncoder, self).__init__()
        # input is 4 x 32 x 32
        self.enc_conv1 = nn.Conv2d(4, chdim, 4, 2, 1)
        # state size. (chdim) x 16 x 16

        self.enc_conv2 = nn.Conv2d(chdim, chdim * 2, 4, 2, 1)
        self.enc_bn2 = nn.BatchNorm2d(chdim * 2)
        # state size. (chdim*2) x 8 x 8

        self.enc_conv3 = nn.Conv2d(chdim * 2, chdim * 4, 4, 2, 1)
        self.enc_bn3 = nn.BatchNorm2d(chdim * 4)
        # state size. (chdim*4) x 4 x 4

        self.dec_conv3 = nn.ConvTranspose2d(chdim * 4, chdim * 2, 4, 2, 1)
        self.dec_bn3 = nn.BatchNorm2d(chdim * 2)
        # state size. (chdim*2) x 8 x 8

        self.dec_conv2 = nn.ConvTranspose2d(chdim * 2, chdim, 4, 2, 1)
        self.dec_bn2 = nn.BatchNorm2d(chdim)
        # state size. (chdim) x 16 x 16

        self.dec_conv1 = nn.ConvTranspose2d(chdim, 4, 4, 2, 1)
        # state size. 4 x 32 x 32

    def forward(self, x):
        out = F.leaky_relu(self.enc_conv1(x), 0.2, True)
        out = F.leaky_relu(self.enc_bn2(self.enc_conv2(out)), 0.2, True)
        out = F.leaky_relu(self.enc_bn3(self.enc_conv3(out)), 0.2, True)
        out = F.leaky_relu(self.dec_bn3(self.dec_conv3(out)), 0.2, True)
        out = F.leaky_relu(self.dec_bn2(self.dec_conv2(out)), 0.2, True)
        out = self.dec_conv1(out)
        return out


"""
    Misc. functions
"""
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def activation_layer(activation):
    # ReLU
    if 'ReLU' == activation:
        return nn.ReLU(inplace=True)
    # LeakyReLU
    elif 'LeakyReLU' == activation:
        return nn.LeakyReLU(inplace=True)
    # ELU
    elif 'ELU' == activation:
        return nn.ELU(inplace=True)
    # PReLU
    elif 'PReLU' == activation:
        return nn.PReLU()
    # RReLU
    elif 'RReLU' == activation:
        return nn.RReLU(inplace=True)
    # SELU
    elif 'SELU' == activation:
        return nn.SELU(inplace=True)
    # Tanh
    elif 'Tanh' == activation:
        return nn.Tanh()
    # others
    else:
        assert False, ("Error: invalid activation type - {}.".format(activation))
    # done.
