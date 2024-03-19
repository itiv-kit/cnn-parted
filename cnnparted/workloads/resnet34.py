# from torchvision import models

from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def _resnet18_cifar(pretrained=False, num_classes=10):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def _resnet34_cifar(pretrained=False, num_classes=10):
    model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def _resnet50_cifar(pretrained=False, num_classes=10):
    model = ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def _resnet101_cifar(pretrained=False, num_classes=10):
    model = ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model

def _resnet152_cifar(pretrained=False, num_classes=10):
    model = ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


resnet18_cifar10 = partial(_resnet18_cifar, num_classes=10)
resnet18_cifar100 = partial(_resnet18_cifar, num_classes=100)
resnet34_cifar10 = partial(_resnet34_cifar, num_classes=10)
resnet34_cifar100 = partial(_resnet34_cifar, num_classes=100)
resnet50_cifar10 = partial(_resnet50_cifar, num_classes=10)
resnet50_cifar100 = partial(_resnet50_cifar, num_classes=100)
resnet101_cifar10 = partial(_resnet101_cifar, num_classes=10)
resnet101_cifar100 = partial(_resnet101_cifar, num_classes=100)
resnet152_cifar10 = partial(_resnet152_cifar, num_classes=10)
resnet152_cifar100 = partial(_resnet152_cifar, num_classes=100)

model = resnet34_cifar100()

accuracy_function = compute_classification_accuracy
