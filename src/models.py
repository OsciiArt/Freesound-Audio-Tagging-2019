import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import pretrainedmodels


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.mode = 'train'

        self.base_model = pretrainedmodels.__dict__['resnet34'](num_classes=num_classes, pretrained=None)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.last_linear = nn.Linear(self.base_model.layer4[1].conv1.in_channels, num_classes)
        self.last_linear = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )
        self.last_linear2 = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, input):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear(x)

        return x

    def noisy(self, input):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.gmp(x4).view(bs, -1)
        x = self.last_linear2(x)

        return x



class ConvBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBnRelu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True))

    def forward(self, x):
        return self.conv_bn_relu(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class EnvNetv2(nn.Module):
    def __init__(self, num_classes=1):
        super(EnvNetv2, self).__init__()
        self.conv1 = ConvBnRelu(1, 32, (1, 64), stride=(1, 2))
        self.conv2 = ConvBnRelu(32, 64, (1, 16), stride=(1, 2))
        self.conv3 = ConvBnRelu(1, 32, (8, 8))
        self.conv4 = ConvBnRelu(32, 32, (8, 8))
        self.conv5 = ConvBnRelu(32, 64, (1, 4))
        self.conv6 = ConvBnRelu(64, 64, (1, 4))
        self.conv7 = ConvBnRelu(64, 128, (1, 2))
        self.conv8 = ConvBnRelu(128, 128, (1, 2))
        self.conv9 = ConvBnRelu(128, 256, (1, 2))
        self.conv10 = ConvBnRelu(256, 256, (1, 2))
        self.maxpool1 = nn.MaxPool2d((1, 64), stride=(1, 64))
        self.maxpool2 = nn.MaxPool2d((5, 3), stride=(5, 3))
        self.maxpool3 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.gmp = nn.AdaptiveMaxPool2d((10, 1))
        self.flatten = Flatten()
        self.last_linear1 = nn.Sequential(
            nn.Linear(256 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )
        self.last_linear2 = nn.Sequential(
            nn.Linear(256 * 10, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, input):
        h = self.conv1(input)
        h = self.conv2(h)
        h = self.maxpool1(h)
        h = h.transpose(1, 2)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.maxpool2(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.maxpool3(h)
        h = self.conv7(h)
        h = self.conv8(h)
        h = self.maxpool3(h)
        h = self.conv9(h)
        h = self.conv10(h)
        h = self.gmp(h)
        h = self.flatten(h)
        h = self.last_linear1(h)
        return h

    def noisy(self, input):
        h = self.conv1(input)
        h = self.conv2(h)
        h = self.maxpool1(h)
        h = h.transpose(1, 2)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.maxpool2(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.maxpool3(h)
        h = self.conv7(h)
        h = self.conv8(h)
        h = self.maxpool3(h)
        h = self.conv9(h)
        h = self.conv10(h)
        h = self.gmp(h)
        h = self.flatten(h)
        h = self.last_linear2(h)
        return h