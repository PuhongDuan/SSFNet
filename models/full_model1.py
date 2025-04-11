import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .__init__ import *
import math
import numpy as np
import cv2
import random
from torch.nn import init

import pandas as pd
dtype = torch.cuda.FloatTensor
from collections import OrderedDict
device = torch.device("cuda:0" )
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        x = x.type(dtype)
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V
class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
class FullModel(nn.Module):
    def __init__(self, arch, n_classes, mode, energy_thr):
        self.mode = mode
        self.energy_thr = energy_thr
        super(FullModel, self).__init__()
        if arch == 'resnet18':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = resnet18(pretrained=True)
            self.base_s2 = resnet18(pretrained=True)
            # self.base_s1.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
            #                            bias=True)
        if arch == 'resnet34':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = resnet34(pretrained=True)
            self.base_s2 = resnet34(pretrained=True)
        elif arch == 'convnext':
            self.feature_dim_s1 = 768
            self.feature_dim_s2 = 768
            self.base_s1 = convnext(pretrained=True)
            self.base_s2 = convnext(pretrained=True)
        elif arch == 'mit':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = mit(pretrained=True)
            self.base_s2 = mit(pretrained=True)
        elif arch == 'resnet50':
            self.feature_dim_s1 = 2048
            self.feature_dim_s2 = 2048
            self.base_s1 = resnet50(pretrained=True)
            self.base_s2 = resnet50(pretrained=True)
        elif arch == 'resnet101':
            self.feature_dim_s1 = 2048
            self.feature_dim_s2 = 2048
            self.base_s1 = resnet101(pretrained=True)
            self.base_s2 = resnet101(pretrained=True)
        elif arch == 'alexnet':
            self.feature_dim_s1 = 256
            self.feature_dim_s2 = 256
            self.base_s1 = alexnet(pretrained=True)
            self.base_s2 = alexnet(pretrained=True)
        elif arch == 'vgg16':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = vgg16_bn(pretrained=True)
            self.base_s2 = vgg16_bn(pretrained=True)
        elif arch == 'googlenet':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = googlenet(pretrained=True)
            self.base_s2 = googlenet(pretrained=True)



        self.sigmoid = nn.Sigmoid()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.pooling_gp = nn.AdaptiveAvgPool2d(1)
        self.fc_s1 = nn.Linear(self.feature_dim_s1, n_classes)
        self.fc_s2 = nn.Linear(self.feature_dim_s2, n_classes)
        self.fc_s3 = nn.Linear(5, n_classes)
        self.gp = nn.Sequential(
            nn.Conv1d(224, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # nn.Linear(512, n_classes)
        )
        self.linear = nn.Sequential(
            nn.Linear(224, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(512, n_classes)
        )
    def clsmap_norm(self, feat_map):
        feat_b, feat_c, feat_h, feat_w = feat_map.size()
        feat_map = feat_map.view(feat_b, feat_c, -1).permute(0,2,1)
        heat_map =self.fc_s1(feat_map).permute(0,2,1)

        heat_map = heat_map.view(feat_b, -1, feat_h, feat_w)
        return heat_map

    def featmap_norm(self, feat_map):
        feat_map = feat_map.sum(dim=1).unsqueeze(dim=1)
        feat_map = F.upsample(feat_map, size=(25, 25), mode='bilinear', align_corners = True).squeeze(dim=1)
        feat_b, feat_h, feat_w = feat_map.size(0), feat_map.size(1), feat_map.size(2)

        feat_map = feat_map.view(feat_map.size(0), -1)
        feat_map_max, _ = torch.max(feat_map, dim=1)
        feat_map_min, _ = torch.min(feat_map, dim=1)
        feat_map_max = feat_map_max.view(feat_b, 1)
        feat_map_min = feat_map_min.view(feat_b, 1)
        feat_map = (feat_map - feat_map_min) / (feat_map_max - feat_map_min)
        feat_map = feat_map.view(feat_b, 1, feat_h, feat_w)
        return feat_map
    def structured_searching(self, feat_vec):
        feat_b, feat_l = feat_vec.size(0), feat_vec.size(1)
        str = np.zeros(shape=feat_b, dtype=int)
        end = np.zeros(shape=feat_b, dtype=int)
        inf_total = feat_vec.sum(dim=1)

        len_init_thr = 0.5
        for i in range(feat_b):
            info_max = 0
            cen = 0
            # search center and inital h and w
            for j in range(int(feat_l*len_init_thr/2), int(feat_l*(1-len_init_thr/2))):
                enrgy_thr = feat_vec[i, j-int(feat_l*len_init_thr/2):j+int(feat_l*len_init_thr/2)].sum() / inf_total[i]
                if  enrgy_thr >= info_max:
                    info_max = enrgy_thr    
                    cen = j
            str[i] = max(cen-int(feat_l*len_init_thr/2), 0)
            end[i] = min(cen+int(feat_l*len_init_thr/2), feat_l)

            #search final h 
            enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
            # print ('energy: ', enrgy_thr)
            # print ('str, end: ', str, end)
            if enrgy_thr < self.energy_thr:
                # print ('+++: ')
                while enrgy_thr < self.energy_thr:
                    if str[i] == 0:
                        end[i] += 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    elif end[i] == feat_l - 1:
                        str[i] -= 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    elif feat_vec[i, str[i]-1] > feat_vec[i, end[i]+1]:
                        str[i] -= 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    elif feat_vec[i, str[i]-1] < feat_vec[i, end[i]+1]:
                        end[i] += 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    # print ('energy: ', enrgy_thr)
                    # print ('str, end: ', str, end)
            else:
                # print ('---: ')
                while enrgy_thr > self.energy_thr:
                    if feat_vec[i, str[i]] > feat_vec[i, end[i]]:
                        end[i] -= 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    else:
                        str[i] += 1
                        enrgy_thr = feat_vec[i, str[i]:end[i]].sum() / inf_total[i]
                    # print ('energy: ', enrgy_thr)
                    # print ('str, end: ', str, end)
        str = str.astype(float)
        end = end.astype(float)
        str = str / (feat_l*1.0)
        end = end / (feat_l*1.0)

        return str, end

    def bounding_box(self, feat_map, is_training):
        feat_map = feat_map.squeeze(dim=1)
        feat_b = feat_map.size(0)
        feat_vec_h = feat_map.sum(dim=2)
        feat_vec_w = feat_map.sum(dim=1)

        if not is_training:
            h_str, h_end = self.structured_searching(feat_vec_h)
            w_str, w_end = self.structured_searching(feat_vec_w)
        else:
            h_str = np.zeros(shape=feat_b, dtype=float)
            h_end = np.zeros(shape=feat_b, dtype=float)
            w_str = np.zeros(shape=feat_b, dtype=float)
            w_end = np.zeros(shape=feat_b, dtype=float)
            for i in range(feat_b):
                h_str[i] = random.uniform(0, 1-0.5)
                h_end[i] = h_str[i] + 0.5
                w_str[i] = random.uniform(0, 1-0.5)
                w_end[i] = w_str[i] + 0.5

        return [h_str, h_end, w_str, w_end]

    def img_sampling(self, img, h_str, h_end, w_str, w_end):
        img_b, img_c, img_h, img_w = img.size()
        img_sampled = torch.zeros(img_b, img_c, int(img_h/4), int(img_w/4)).cuda()
        h_str = (h_str*img_h).astype(int)
        h_end = (h_end*img_h).astype(int)
        w_str = (w_str*img_w).astype(int)
        w_end = (w_end*img_w).astype(int)
        for i in range(img_b):
            img_sampled_i = img[i, :, h_str[i]:h_end[i], w_str[i]:w_end[i]].unsqueeze(dim=0)
            img_sampled[i, :] = F.upsample(img_sampled_i, size=(int(img_h/4), int(img_w/4)), mode='bilinear', align_corners = True)

        return img_sampled
 
    def get_parameters(self):
        if self.mode == 's1':
            for i in self.base_s2.parameters():
                i.requires_grad = False
            for i in self.fc_s2.parameters():
                i.requires_grad = False
            params = list(self.base_s1.parameters()) + list(self.fc_s1.parameters()) 
        elif self.mode == 's2':
            for i in self.base_s1.parameters():
                i.requires_grad = False
            for i in self.fc_s1.parameters():
                i.requires_grad = False
            params = list(self.base_s2.parameters()) + list(self.fc_s2.parameters())
        return params
 
    def baseline_searching(self, feat_vec):
        feat_b, feat_l = feat_vec.size(0), feat_vec.size(1)

        str = np.zeros(shape=feat_b, dtype=int)
        end = np.zeros(shape=feat_b, dtype=int)

        for i in range(feat_b):
            for j in range(feat_l):
                if feat_vec[i, j] != 0:
                    if str[i]==0 and j>0:
                        str[i] = j-1
                    elif str[i]==0 and j==0:
                        str[i] =0
                    end[i] = j


        str = str.astype(float)
        end = end.astype(float)
        str = str / (feat_l*1.0)
        end = end / (feat_l*1.0)

        return str, end

    def baseline_bounding_box(self, feat_map, is_training):
        feat_map = (feat_map - 0.7) + 1
        feat_map = feat_map.int().float()

        feat_map = feat_map.squeeze(dim=1)
        feat_b = feat_map.size(0)
        feat_vec_h = feat_map.sum(dim=2)
        feat_vec_w = feat_map.sum(dim=1)

        if not is_training:
            h_str, h_end = self.baseline_searching(feat_vec_h)
            w_str, w_end = self.baseline_searching(feat_vec_w)
        else:
            h_str = np.zeros(shape=feat_b, dtype=float)
            h_end = np.zeros(shape=feat_b, dtype=float)
            w_str = np.zeros(shape=feat_b, dtype=float)
            w_end = np.zeros(shape=feat_b, dtype=float)
            for i in range(feat_b):
                h_str[i] = random.uniform(0, 1-0.5)
                h_end[i] = h_str[i] + 0.5
                w_str[i] = random.uniform(0, 1-0.5)
                w_end[i] = w_str[i] + 0.5

        return [h_str, h_end, w_str, w_end]


    def forward(self, img_s1, img_s2, labels, is_training=True):
        img_s1 = img_s1.type(dtype)
        img_s2 = img_s2.type(dtype)
        img_gp = img_s2
        img_b = img_s1.size(0)
        heat_map, [h_str, h_end, w_str, w_end] = None, [None, None, None, None]
        # feat_map_s2 = self.gp(img_s2)
        # feat_map_s2 = torch.mean(feat_map_s2,2)
        # img_s2 = img_s2.view(img_s2.size(0), -1)
        # feat_map_s2 = self.linear(img_s2)
        if self.mode == 's1':
            feat_map_s1 = self.base_s1(img_s1)  #.get_features
            # model = SKAttention(channel=512,reduction=8)
            # model = model.to(device)
            # feat_map_s1 = model(feat_map_s1)
            feat_map_s1 = self.pooling(feat_map_s1[3]).view(img_s1.size(0), -1)
            # feat_map_s2 = self.base_s2.get_features(img_s2)
            # feat_map_s2 = self.pooling(feat_map_s2).view(img_s2.size(0), -1)
            # feat_map = torch.cat([feat_map_s1, feat_map_s2], axis=1)
            # logits_s1 = self.fc_s1(feat_map_s1 + feat_map_s2)
            # logits_s2 = self.fc_s1(feat_map_s2)
            logits_s1 = self.fc_s1(feat_map_s1) #  + feat_map_s2
            # logits_s1 = self.fc_s1(feat_map)
            logits = logits_s1 #+ logits_s2#* (1 + img_s2)

        elif self.mode == 's2':
            with torch.no_grad():
                # feat_map_s1 = self.base_s1.get_features(img_s1)
                # logits_s1 = self.fc_s1(self.pooling(feat_map_s1).view(img_b, -1))
                feat_map_s1 = self.base_s1(img_s1)
                feat_map_s1 = self.pooling(feat_map_s1[3]).view(img_s1.size(0), -1)
                heat_map = self.featmap_norm(feat_map_s1)
                h_str, h_end, w_str, w_end = self.bounding_box(heat_map, is_training)
                # h_str, h_end, w_str, w_end = self.baseline_bounding_box(heat_map, is_training)
                img_s2 = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
                img_s2 = F.upsample(img_s2, size=(int(224), int(224)), mode='bilinear', align_corners=True)
            # img_zs = img_s2.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            # img_zs = img_zs.view(img_s2.size(0), -1)
            # img_sum = torch.sum(img_zs, axis=1)
            # for k in range(img_zs.size(0)):
            #     img_zs[k, :] = img_zs[k, :]/img_sum[k]
            # logits_s2 = self.fc_s3(img_zs)
            # img_gp = img_s2.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            img_gp = img_gp.view(img_gp.size(0), -1)
            logits_gp = self.linear(img_gp)
            feat_map_s2 = self.base_s2.get_features(img_s2)
            logits_s2 = self.fc_s2(self.pooling(feat_map_s2).view(img_b, -1) + logits_gp) #+ self.pooling(feat_map_s1).view(img_b, -1)

            if is_training:
                logits = logits_s2
            else:
                logits = logits_s1 + logits_s2

        elif self.mode == 's3':
            with torch.no_grad():
                feat_s3 = self.net(img_s1).detach().cpu().numpy()
                X = self.pca.fit_transform(feat_s3)
                Y = labels
                logits = self.svm.fit(X, Y)

        return logits, [h_str, h_end, w_str, w_end], heat_map
    # def forward(self, img_s1, img_s2, is_training=True):
    #     img_s1 = img_s1.type(dtype)
    #     img_s2 = img_s2.type(dtype)
    #     img_b = img_s1.size(0)
    #     heat_map, [h_str, h_end, w_str, w_end] = None, [None, None, None, None]
    #     # feat_gp = self.gp(img_s2)
    #     feat_gp = img_s2.view(img_s2.size(0), -1)
    #     feat_gp = self.linear(feat_gp) #.view(img_s2.size(0), -1)
    #     if self.mode == 's1':
    #         feat_map_s1 = self.base_s1.get_features(img_s1)
    #         feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
    #         feat_map = torch.cat([feat_map_s1, feat_gp], axis=1)
    #         logits_s1 = self.fc_s1(feat_map)
    #         # logits_s1 = self.fc_s1(feat_map_s1) + feat_gp
    #         # logits_s1 = self.fc_s1(feat_map_s1 + feat_gp)
    #         logits = logits_s1
# class Trainer:
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#         self._init_model()
#
#     def _init_model(self):
#         self.net = Model(self.args).to(self.device)
#         self.opt = torch.optim.Adam(self.net.parameters(), )
#         self.svm = SVC(kernel='rbf')
#         self.pca = PCA(n_components=self.args.K)
#         # self.pca = manifold.TSNE(n_components=self.args.K, init='pca')
#
#
#     def feature_extract(self):
#         outputs = []
#         labels = []
#         print("进行特征提取...")
#         for inputs, targets in tqdm(self.dl, ncols=90):
#             inputs = inputs.to(self.device)
#             targets = targets.numpy()
#             output = self.net(inputs).detach().cpu().numpy()
#             outputs.append(output)
#             labels.append(targets)
#
#         X = np.concatenate(outputs, axis=0)
#         y = np.concatenate(labels, axis=0)
#
#         data = {'X': X, 'y': y}
#         io.savemat('results/%s.mat' % self.args.dataset, data)
#
#     def train(self):
#         print("数据集: ", self.args.dataset)
#         print("train ratio: ", self.args.ratio)
#
#         print("读取数据集...")
#         data = io.loadmat('results/%s.mat' % self.args.dataset)
#         X, y = data['X'], data['y'].squeeze()
#         print("pca降维...")
#         X = self.pca.fit_transform(X)
#         print("划分数据集...")
#         X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.args.ratio)
#         self.svm.fit(X_train, y_train)
#         pred = self.svm.predict(X_test)
#         acc = accuracy_score(y_test, pred)
#         print('val_acc: %.6f' % acc)
#
#     def forward(self, img_s1, img_s2, is_training=True):
#         img_s1 = img_s1.type(dtype)
#         img_s2 = img_s2.type(dtype)
#         feat = self.net(img_s1).detach().cpu().numpy()
#         feat_pca = self.pca.fit_transform(feat)
#         self.svm.fit(X_train, y_train)