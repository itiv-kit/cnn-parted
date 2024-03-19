# from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.avg_pool2d = nn.AvgPool2d(2)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _mobilenet_cifar(pretrained=False, num_classes=10):
    model = MobileNet(num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


mobilenet_cifar10 = partial(_mobilenet_cifar, num_classes=10)
mobilenet_cifar100 = partial(_mobilenet_cifar, num_classes=100)

model = mobilenet_cifar100()

accuracy_function = compute_classification_accuracy
