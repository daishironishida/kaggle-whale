from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M

from .utils import ON_KAGGLE


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


def create_net(net_cls, pretrained: bool):
    if ON_KAGGLE and pretrained:
        net = net_cls()
        model_name = net_cls.__name__
        weights_path = f'../input/{model_name}/{model_name}.pth'
        net.load_state_dict(torch.load(weights_path))
    else:
        net = net_cls(pretrained=pretrained)
    return net


class ResNet(nn.Module):
    def __init__(self, num_features, head_mid=16,
                 pretrained=False, net_cls=M.resnet101, dropout=0):
        super().__init__()
        self.num_features = num_features
        self.head_mid = head_mid

        self.base = create_net(net_cls, pretrained=pretrained)
        self.base.avgpool = AvgPool()
        if dropout:
            self.base.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.base.fc.in_features, num_features),
            )
        else:
            self.base.fc = nn.Linear(self.base.fc.in_features, num_features)

        self.head_conv = nn.Conv1d(4, head_mid, 1)
        self.head_fc = nn.Linear(self.num_features * self.head_mid, 2)

    def fresh_params(self):
        return self.base.fc.parameters()

    def forward_head(self, feature1, feature2):
        x1 = feature1 * feature2
        x2 = feature1 + feature2
        x3 = abs(feature1 - feature2)
        x4 = x3 * x3

        features = torch.cat((x1, x2, x3, x4), dim=1).view(-1, 4, self.num_features)
        features = F.relu(self.head_conv(features))
        return self.head_fc(features.view(-1, self.num_features * self.head_mid))

    def forward_once(self, x):
        return F.relu(self.base(x))

    def forward(self, input1, input2):
        feature1 = self.forward_once(input1)
        feature2 = self.forward_once(input2)
        return torch.softmax(self.forward_head(feature1, feature2), dim=1)


resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)