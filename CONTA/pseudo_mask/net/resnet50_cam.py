import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50

class Net_50(nn.Module):

    def __init__(self):
        super(Net_50, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 103, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 103)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM_50(Net_50):

    def __init__(self):
        super(CAM_50, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x


class Net_101(nn.Module):

    def __init__(self):
        super(Net_101, self).__init__()

        self.resnet50 = resnet50.resnet101(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)

        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.classifier = nn.Conv2d(2048, 103, 1, bias=False)

        # self.stage4_e1 = nn.Sequential(self.resnet50.layer4)
        # self.stage4_e2 = nn.Sequential(self.resnet50.layer4)
        # self.stage4_e3 = nn.Sequential(self.resnet50.layer4)
        #
        # self.classifier_e1 = nn.Conv2d(2048, 103, 1, bias=False)
        # self.classifier_e2 = nn.Conv2d(2048, 103, 1, bias=False)
        # self.classifier_e3 = nn.Conv2d(2048, 103, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)
        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 103)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM_101(Net_101):

    def __init__(self):
        super(CAM_101, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x


class Net_101_ENSEMBLE(nn.Module):

    def __init__(self):
        super(Net_101_ENSEMBLE, self).__init__()

        self.resnet50 = resnet50.resnet101(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)

        # self.stage4 = nn.Sequential(self.resnet50.layer4)
        # self.classifier = nn.Conv2d(2048, 103, 1, bias=False)

        self.stage4_e1 = nn.Sequential(self.resnet50.layer4)
        self.stage4_e2 = nn.Sequential(self.resnet50.layer4)
        self.stage4_e3 = nn.Sequential(self.resnet50.layer4)

        self.classifier_e1 = nn.Conv2d(2048, 103, 1, bias=False)
        self.classifier_e2 = nn.Conv2d(2048, 103, 1, bias=False)
        self.classifier_e3 = nn.Conv2d(2048, 103, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4_e1, self.stage4_e2, self.stage4_e3])
        self.newly_added = nn.ModuleList([self.classifier_e1, self.classifier_e2, self.classifier_e3])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()
        x = self.stage3(x)

        x1 = self.stage4_e1(x)
        x1 = torchutils.gap2d(x1, keepdims=True)
        x1 = self.classifier_e1(x1)
        x1 = x1.view(-1, 103)

        x2 = self.stage4_e2(x)
        x2 = torchutils.gap2d(x2, keepdims=True)
        x2 = self.classifier_e2(x2)
        x2 = x2.view(-1, 103)

        x3 = self.stage4_e3(x)
        x3 = torchutils.gap2d(x3, keepdims=True)
        x3 = self.classifier_e3(x3)
        x3 = x3.view(-1, 103)

        return [x1, x2, x3]


    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class CAM_101_ENSEMBLE(Net_101_ENSEMBLE):

    def __init__(self):
        super(CAM_101_ENSEMBLE, self).__init__()

    def forward(self, x):

        x = self.stage1(x).detach()
        x = self.stage2(x).detach()
        x = self.stage3(x).detach()

        x1 = self.stage4_e1(x).detach()
        cam1 = F.conv2d(x1, self.classifier_e1.weight).detach()
        cam1 = F.relu(cam1).detach()
        cam1 = cam1[0] + cam1[1].flip(-1).detach()

        x2 = self.stage4_e2(x).detach()
        cam2 = F.conv2d(x2, self.classifier_e2.weight).detach()
        cam2 = F.relu(cam2).detach()
        cam2 = cam2[0] + cam2[1].flip(-1).detach()

        x3 = self.stage4_e3(x).detach()
        cam3 = F.conv2d(x3, self.classifier_e3.weight).detach()
        cam3 = F.relu(cam3).detach()
        cam3 = cam3[0] + cam3[1].flip(-1)

        cam = cam1 + cam2 + cam3

        return cam

class Net_152(nn.Module):

    def __init__(self):
        super(Net_152, self).__init__()

        self.resnet50 = resnet50.resnet152(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.classifier = nn.Conv2d(2048, 103, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 103)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class CAM_152(Net_152):

    def __init__(self):
        super(CAM_152, self).__init__()

    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
