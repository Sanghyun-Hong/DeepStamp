"""
    AlexNet architecture for CIFAR10.
"""
# torch
import torch.nn as nn
import torch.nn.init
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable


__all__ = ['AlexNet', 'alexnet']


class AlexNetBase(nn.Module):
    def __init__(self, activation, num_classes=10):
        super(AlexNetBase, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            activation(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            activation(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 128),
            activation(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            activation(inplace=True),
            nn.Linear(128, num_classes),
        )

        # define activation (default: ReLU)
        self.activation_name = 'elu' if activation == nn.ELU else 'relu'

        # init. layer: xavier_normal_
        #self.apply(self.init_layer)


    def init_layer(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            gain = nn.init.calculate_gain('relu') if self.activation_name == 'relu' else 1.0
            nn.init.xavier_normal_(m.weight, gain=gain)


    def forward(self, x, early_also=False):
        x = self.features(x)
        f = x.view(x.size(0), 256 * 1 * 1)
        x = self.classifier(f)
        return (x, f) if early_also else x


def Cifar10_AlexNet(**kwargs):
    return AlexNetBase(activation=nn.ReLU, **kwargs)


"""
    Main (test purpose)
"""
if __name__ == '__main__':
    net  = Cifar10_AlexNet()
    data = torch.randn(1,3,32,32)
    out  = net(data)
    print (out)
    # Fin.
