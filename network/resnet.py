import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from network.attention import DAM


class RawResNet(nn.Module):
    def __init__(self, arch='resnet34', in_channels=3, out_channels=5, n_filters=64):
        super(RawResNet, self).__init__()
        if arch == 'resnet18':
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == 'resnet34':
            self.net = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif arch == 'resnet50':
            self.net = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.net.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, out_channels)

    def forward(self, x):
        return self.net(x)


class SpectrogramResNet(nn.Module):
    def __init__(self, arch='resnet34', in_channels=3, out_channels=5, n_filters=64):
        super(SpectrogramResNet, self).__init__()
        if arch == 'resnet18':
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == 'resnet34':
            self.net = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif arch == 'resnet50':
            self.net = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.net.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, out_channels)

    def forward(self, x):
        return self.net(x)


class RawAttResNet(nn.Module):
    def __init__(self, arch='resnet34', in_channels=3, out_channels=5, n_filters=64):
        super(RawAttResNet, self).__init__()
        if arch == 'resnet18':
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == 'resnet34':
            self.net = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif arch == 'resnet50':
            self.net = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.net.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.body = nn.Sequential(*list(self.net.children())[:-2])
        self.flatten = nn.Flatten()
        inter_channels = self.net.fc.in_features
        self.head = DAM(inter_channels, inter_channels, inter_channels // 32)
        self.fc = nn.Linear(inter_channels, out_channels, bias=False)

    def forward(self, x):
        b, _, h, w = x.size()
        out = self.body(x)
        out = self.head(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class SpectrogramAttResNet(nn.Module):
    def __init__(self, arch='resnet34', in_channels=3, out_channels=5, n_filters=64):
        super(SpectrogramAttResNet, self).__init__()
        if arch == 'resnet18':
            self.net = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif arch == 'resnet34':
            self.net = resnet34(pretrained=ResNet34_Weights.DEFAULT)
        elif arch == 'resnet50':
            self.net = resnet50(pretrained=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError
        self.net.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_filters, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.body = nn.Sequential(*list(self.net.children())[:-2])
        self.flatten = nn.Flatten()
        inter_channels = self.net.fc.in_features
        self.head = DAM(inter_channels, inter_channels, inter_channels // 4)
        self.fc = nn.Linear(inter_channels * 6, out_channels, bias=False)  # for seg1500 down0
        self.fc = nn.Linear(inter_channels * 4, out_channels, bias=False) # for seg1000 down0
        # self.fc = nn.Linear(inter_channels, out_channels, bias=False) # for seg256 down0, seg1024 down4,

    def forward(self, x):
        out = self.body(x)
        out = self.head(out)
        # print(out.shape)
        out = self.flatten(out)
        out = self.fc(out)
        return out
