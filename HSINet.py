import torch
import torch.nn as nn
from fightingcv_attention.attention.CBAM import *
from fightingcv_attention.attention.ResidualAttention import ResidualAttention
from fightingcv_attention.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
from fightingcv_attention.attention.PSA import PSA
# from ScConv import *

class HSIFeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(HSIFeatureExtractor, self).__init__()

        self.cbam1 = CBAMBlock(channel=256, reduction=16, kernel_size=7)
        self.cbam2 = CBAMBlock(channel=512, reduction=16, kernel_size=7)
        self.cbam3 = CBAMBlock(channel=1024, reduction=16, kernel_size=7)
        self.cbam4 = CBAMBlock(channel=2048, reduction=16, kernel_size=7)
        self.cbam5 = CBAMBlock(channel=128, reduction=16, kernel_size=7)

        self.resatt1 = ResidualAttention(channel=256, num_class=11)
        self.resatt2 = ResidualAttention(channel=512, num_class=11)
        self.resatt3 = ResidualAttention(channel=1024, num_class=11)

        # self.psa1 = SequentialPolarizedSelfAttention(channel=256)
        # self.psa2 = SequentialPolarizedSelfAttention(channel=512)
        # self.psa3 = SequentialPolarizedSelfAttention(channel=1024)
        self.psa = PSA(channel=256, reduction=4)

        #ScConv
        # self.sc1 = ScConv(224)
        # self.sc2 = ScConv(256)
        # self.sc3 = ScConv(512)
        # self.sc4 = ScConv(1024)
        # self.sc5 = ScConv(2048)

        #1*1卷积层
        self.c1 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=1)
        self.c2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.c3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.c4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.c5 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1)

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=224, out_channels=256, stride=1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, stride=1, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(in_channels=1024, out_channels=2048, stride=1, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc1 = nn.Linear(2048 * (height // 16) * (width // 16), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(2048 * (height // 16) * (width // 16), 128)
        self.fc4 = nn.Linear(128, num_classes)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x1 = self.c1(x)
        # x1 = self.sc1(x1)
        # x1 = self.c1(x1)
        # x = x + x1
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.pool(x)
        # x = self.cbam1(x)
        # x = self.psa(x)
        # x2 = self.c2(x)
        # x2 = self.sc2(x2)
        # x2 = self.c2(x2)
        # x = x + x2
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.bn2(x)
        x = self.pool(x)
        # x = self.cbam2(x)
        # x3 = self.c3(x)
        # x3 = self.sc3(x3)
        # x3 = self.c3(x3)
        # x = x + x3
        x = self.conv3(x)
        x = self.relu(x)
        # x = self.bn3(x)
        x = self.pool(x)
        # x = self.cbam3(x)
        # x4 = self.c4(x)
        # x4 = self.sc4(x4)
        # x4 = self.c4(x4)
        # x = x + x4
        x = self.conv4(x)
        x = self.relu(x)
        # x = self.bn4(x)
        x = self.pool(x)
        x5 = self.c5(x)
        x5 = self.sc5(x5)
        x5 = self.c5(x5)
        x = x + x5
        # x = self.cbam4(x)
        # x = self.relu(self.conv5(x))
        # x = self.pool(x)
        # x = self.cbam5(x)
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # def forward(self, x):
    #     x = self.relu(self.conv1(x))
    #     x = self.pool(x)
    #     # x = self.cbam5(x)
    #     xc, xs = self.cbam1(x)
    #     # x = self.psa1(x)
    #     xc = self.relu(self.conv2(xc))
    #     xs = self.relu(self.conv2(xs))
    #     xc = self.pool(xc)
    #     xs = self.pool(xs)
    #     xc_1, xs_1 = self.cbam2(xc)
    #     xc_2, xs_2 = self.cbam2(xs)
    #     xc = xc_1 + xc_2
    #     xs = xs_1 + xs_2
    #     # x = self.psa2(x)
    #     xc = self.relu(self.conv3(xc))
    #     xs = self.relu(self.conv3(xs))
    #     xc = self.pool(xc)
    #     xs = self.pool(xs)
    #     xc_1, xs_1 = self.cbam3(xc)
    #     xc_2, xs_2 = self.cbam3(xs)
    #     xc = xc_1 + xc_2
    #     xs = xs_1 + xs_2
    #     # x = self.psa3(x)
    #     xc = self.relu(self.conv4(xc))
    #     xs = self.relu(self.conv4(xs))
    #     xc = self.pool(xc)
    #     xs = self.pool(xs)
    #     xc_1, xs_1 = self.cbam4(xc)
    #     xc_2, xs_2 = self.cbam4(xs)
    #     xc = xc_1 + xc_2
    #     xs = xs_1 + xs_2
    #     # x = self.relu(self.conv5(x))
    #     # x = self.pool(x)
    #     # x = self.cbam5(x)
    #     # x = x.view(x.size(0), -1)
    #     xc = xc.view(xc.size(0), -1)
    #     xs = xs.view(xs.size(0), -1)
    #     xc = self.relu(self.fc1(xc))
    #     xc = self.fc2(xc)
    #     xs = self.relu(self.fc3(xs))
    #     xs = self.fc4(xs)
    #     return xc+xs

# input=torch.randn(50,512,7,7)
# kernel_size=input.shape[2]
# cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
# output=cbam(input)
# print(output.shape)
# 创建模型实例
channels = 224  # 高光谱图像通道数
height = 256  # 图像高度
width = 256  # 图像宽度
num_classes = 11  # 分类数
model = HSIFeatureExtractor(num_classes)
# 
# # 打印模型结构
# print(model)
# net = HSIFeatureExtractor(num_classes=11)
# x = torch.randn(16, 224, 256, 256)
# y = net(x)
# print(y.size())
# from fightingcv_attention.attention.PSA import PSA
# input=torch.randn(16,224,256,256)
# psa = PSA(channel=224,reduction=4)
# output=psa(input)
# print(output.shape)