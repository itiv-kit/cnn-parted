# from torchvision import models
from model_explorer.accuracy_functions.classification_accuracy import compute_classification_accuracy

import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def _vgg11_cifar(pretrained=False, num_classes=10):
    model = VGG('VGG11', num_classes)
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


def _vgg13_cifar(pretrained=False, num_classes=10):
    model = VGG('VGG13', num_classes)
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


def _vgg16_cifar(pretrained=False, num_classes=10):
    model = VGG('VGG16', num_classes)
    if pretrained:
        checkpoint = torch.load(pretrained)
        state_dict=checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


def _vgg19_cifar(pretrained=False, num_classes=10):
    model = VGG('VGG19', num_classes)
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        state_dict = checkpoint['net']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]    # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        # model.load_state_dict(m['net'], strict=False)
    return model


vgg11_cifar10 = partial(_vgg11_cifar, num_classes=10)
vgg11_cifar100 = partial(_vgg11_cifar, num_classes=100)
vgg13_cifar10 = partial(_vgg13_cifar, num_classes=10)
vgg13_cifar100 = partial(_vgg13_cifar, num_classes=100)
vgg16_cifar10 = partial(_vgg16_cifar, num_classes=10)
vgg16_cifar100 = partial(_vgg16_cifar, num_classes=100)
vgg19_cifar10 = partial(_vgg19_cifar, num_classes=10)
vgg19_cifar100 = partial(_vgg19_cifar, num_classes=100)

model = vgg11_cifar10()

accuracy_function = compute_classification_accuracy
