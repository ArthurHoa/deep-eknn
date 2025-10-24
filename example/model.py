
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np

def encode_input(trainloader, testloader, model, device='cpu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    train_x_list, train_y_list = [], []
    test_x_list, test_y_list = [], []

    # --- Train reduction ---
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():  # Pas besoin de calculer les gradients
            layer4 = model.getlayer4(inputs)
        
        # Move to CPU before converting to numpy
        features = layer4[:, :, 0, 0].detach().cpu().numpy()
        train_x_list.append(features)
        train_y_list.append(targets.cpu().numpy())

    # --- Test reduction ---
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            layer4 = model.getlayer4(inputs)

        features = layer4[:, :, 0, 0].detach().cpu().numpy()
        test_x_list.append(features)
        test_y_list.append(targets.cpu().numpy())

    train_x = np.concatenate(train_x_list, axis=0, dtype=float)
    train_y = np.concatenate(train_y_list, axis=0, dtype=float)
    test_x = np.concatenate(test_x_list, axis=0, dtype=float)
    test_y = np.concatenate(test_y_list, axis=0, dtype=float)

    return train_x, train_y, test_x, test_y

def load_ResNet18():
    net = ResNet18()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    print('==> Loading pre-trained ResNet18...')
    assert os.path.isdir('example/saved_model'), 'Error: no model found!'
    checkpoint = torch.load('./example/saved_model/model.pth')
    net.load_state_dict(checkpoint['net'])

    net.eval()
    return net

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def getlayer4(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])
