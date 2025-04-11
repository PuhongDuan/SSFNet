import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from HSINet import HSIFeatureExtractor
# import MF2C.MF2CNet
from .__init__ import *
import math
import numpy as np
import cv2
import random
import pandas as pd
dtype = torch.cuda.FloatTensor
# device = torch.device("cuda:0")
from fightingcv_attention.attention.SEAttention import *
from fightingcv_attention.attention.ResidualAttention import *
from fightingcv_attention.attention.ECAAttention import ECAAttention
from absvit import *
# from MF2C import *
# from ScConv import *
# from NetworksBlocks import *
from data import DataLoader, Dataset
from entropy import *

class MinimumClassConfusionLoss(nn.Module):
    r"""
    Minimum Class Confusion loss minimizes the class confusion in the target predictions.

    You can see more details in `Minimum Class Confusion for Versatile Domain Adaptation (ECCV 2020) <https://arxiv.org/abs/1912.03699>`_

    Args:
        temperature (float) : The temperature for rescaling, the prediction will shrink to vanilla softmax if
          temperature is 1.0.

    .. note::
        Make sure that temperature is larger than 0.

    Inputs: g_t
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`

    Shape:
        - g_t: :math:`(minibatch, C)` where C means the number of classes.
        - Output: scalar.

    Examples::
        >>> temperature = 2.0
        >>> loss = MinimumClassConfusionLoss(temperature)
        >>> # logits output from target domain
        >>> g_t = torch.randn(batch_size, num_classes)
        >>> output = loss(g_t)

    MCC can also serve as a regularizer for existing methods.
    Examples::
        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> temperature = 2.0
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim, hidden_size=1024)
        >>> cdan_loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> mcc_loss = MinimumClassConfusionLoss(temperature)
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> total_loss = cdan_loss(g_s, f_s, g_t, f_t) + mcc_loss(g_t)
    """

    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss
    
class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        # self.iter = enumerate(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            # self.iter = enumerate(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class FullModel(nn.Module):
    def __init__(self, arch, n_classes, mode, energy_thr):
        self.mode = mode
        self.arch = arch
        self.energy_thr = energy_thr
        super(FullModel, self).__init__()

        # 1*1卷积层
        self.c1 = nn.Conv2d(in_channels=224, out_channels=224, kernel_size=1)
        self.c2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.c3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.c4 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.c5 = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1)
        self.linear1 = nn.Sequential(
            nn.Linear(1120, 224),
            nn.ReLU(),
            nn.BatchNorm1d(224))
        if arch == 'resnet18':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = resnet18(pretrained=True)
            self.base_s2 = resnet18(pretrained=True)
            self.linear = nn.Sequential(
                nn.Linear(224, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512))
            # self.base_s2.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
            #                            bias=True)
        if arch == 'resnet34':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = resnet34(pretrained=True)
            self.base_s2 = resnet34(pretrained=True)
        elif arch == 'HSINet':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = HSIFeatureExtractor(n_classes)
            self.base_s2 = HSIFeatureExtractor(n_classes)
        elif arch == 'convnext':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = convnext(pretrained=True)
            self.base_s2 = convnext(pretrained=True)
        elif arch == 'mit':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = mit(pretrained=True)
            self.base_s2 = mit(pretrained=True)
            self.linear = nn.Sequential(
                nn.Linear(224, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512))
        elif arch == 'swin_tfmer':
            self.feature_dim_s1 = 768
            self.feature_dim_s2 = 768
            self.base_s1 = swin_tfmer(pretrained=True)
            self.base_s2 = swin_tfmer(pretrained=True)
        elif arch == 'mobile_vit':
            self.feature_dim_s1 = 640
            self.feature_dim_s2 = 640
            self.base_s1 = mobile_vit_small(pretrained=True)
            self.base_s2 = mobile_vit_small(pretrained=True)
        elif arch == 'resnet50':
            self.feature_dim_s1 = 2048
            self.feature_dim_s2 = 2048
            self.base_s1 = resnet50(pretrained=True)
            self.base_s2 = resnet50(pretrained=True)
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
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.BatchNorm1d(2048))
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
            self.linear = nn.Sequential(
                nn.Linear(224, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256))
        elif arch == 'vgg16':
            self.feature_dim_s1 = 512
            self.feature_dim_s2 = 512
            self.base_s1 = vgg16_bn(pretrained=True)
            self.base_s2 = vgg16_bn(pretrained=True)
            self.linear = nn.Sequential(
                nn.Linear(224, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512))
        elif arch == 'googlenet':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = googlenet(pretrained=True)
            self.base_s2 = googlenet(pretrained=True)
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
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024))
        elif arch == 'mixmodel':
            self.feature_dim_s1 = 1024
            self.feature_dim_s2 = 1024
            self.base_s1 = resnet18(pretrained=True)
            self.base_s2 = mit(pretrained=True)
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
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024)
            )
        elif arch == 'absvit':
            self.feature_dim_s1 = 768
            self.feature_dim_s2 = 768
            self.base_s1 = absvit_base_patch16_224(pretrained=True)
            self.base_s2 = absvit_base_patch16_224(pretrained=True)
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
                # nn.BatchNorm1d(1024)
            )
        elif arch == 'vit_ae':
            self.feature_dim_s1 = 768
            self.feature_dim_s2 = 768
            self.base_s1 = ViTAE_basic_Base(pretrained=True, num_classes=11)
            self.base_s2 = ViTAE_basic_Base(pretrained=True, num_classes=11)


        self.sigmoid = nn.Sigmoid()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.pooling_gp = nn.AdaptiveAvgPool2d(1)
        self.fc_s1 = nn.Linear(self.feature_dim_s1, n_classes)
        self.fc_s2 = nn.Linear(self.feature_dim_s2, n_classes)
        self.fc_s3 = nn.Linear(self.feature_dim_s2*2, n_classes)

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

    def featmap_norm1(self, feat_map):
        feat_map = feat_map.sum(dim=1).unsqueeze(dim=1)
        feat_map = F.upsample(feat_map, size=(256, 256), mode='bilinear', align_corners=True).squeeze(dim=1)
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
    def structured_searching1(self, feat_vec):
        feat_b, feat_l = feat_vec.size(0), feat_vec.size(1)
        str = np.zeros(shape=feat_b, dtype=int)
        end = np.zeros(shape=feat_b, dtype=int)
        inf_total = feat_vec.sum(dim=1)
        max_ind = torch.max(feat_vec, axis=1)
        
        for i in range(feat_b):
            info_max = 0
            cen = 0
            start = int(max_ind.indices[i])
            end1 = start + 1
            end2 =  start - 1
            if feat_vec[i, start:end1].sum() > feat_vec[i, end2:start].sum():
                end_ind = end1
            else:
                end_ind = start
                start = start - 1
            enrgy_thr = feat_vec[i, start:end_ind].sum() / inf_total[i]
            while enrgy_thr < self.energy_thr and start < end_ind:
                if feat_vec[i, start-1:end_ind].sum() > feat_vec[i, start:end_ind+1].sum():
                    start = start - 1
                else:
                    end_ind = end_ind + 1
                enrgy_thr = feat_vec[i, start:end_ind].sum() / inf_total[i]
                    
            str[i] = max(start, 0)
            end[i] = min(end_ind, feat_l)
 
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
        img_sampled = torch.zeros(img_b, img_c, img_h, img_w).cuda()
        h_str = (h_str*img_h).astype(int)
        h_end = (h_end*img_h).astype(int)
        w_str = (w_str*img_w).astype(int)
        w_end = (w_end*img_w).astype(int)
        for i in range(img_b):
            img_sampled_i = img[i, :, h_str[i]:h_end[i], w_str[i]:w_end[i]].unsqueeze(dim=0)
            img_sampled[i, :] = F.upsample(img_sampled_i, size=(img_h, img_w), mode='bilinear', align_corners = True)

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
    # def forward(self, img, is_training=True):
    #     img_s1 = img[:,[19,13,7],:,:].type(dtype)  # 256*256
    #     img_s2 = img.type(dtype)
    #     # img_s1 = img_s1[:, :, 16:240, 16:240].type(dtype)  # 224*224
    #     # img_s2 = img_s2[:, :, 16:240, 16:240].type(dtype)
    #     img_b = img_s1.size(0)
    #     heat_map, [h_str, h_end, w_str, w_end] = None, [None, None, None, None]
    #     if self.mode == 's1':
    #         if self.arch == 'mit':
    #             feat_map_s1 = self.base_s1(img_s1)
    #             heat_map = self.featmap_norm(feat_map_s1[3])
    #             feat_map_s1 = self.pooling(feat_map_s1[3]).view(img_s1.size(0), -1)
    #         elif self.arch == 'resnet18':
    #             feat_map_s1 = self.base_s1.get_features(img_s1)
    #             # f3 = self.c3(feat_map_s1[0])
    #             # f3 = self.sc3(f3)
    #             # f3 = self.c3(f3)
    #             # feat_map_s1sc = feat_map_s1[0] + f3
    #             heat_map = self.featmap_norm(feat_map_s1[0])
    #             feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
    #             # heat_map = self.featmap_norm(feat_map_s1[0])
    #             # feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
    #         elif self.arch == 'googlenet':
    #             feat_map_s1 = self.base_s1.get_features(img_s1)
    #             heat_map = self.featmap_norm(feat_map_s1[0])
    #             feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
    #         elif self.arch == 'resnet50':
    #             feat_map_s1 = self.base_s1.get_features(img_s1)
    #             heat_map = self.featmap_norm(feat_map_s1[0])
    #             feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
    #         elif self.arch == 'vgg16':
    #             feat_map_s1 = self.base_s1.get_features(img_s1)
    #             heat_map = self.featmap_norm(feat_map_s1)
    #             feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
    #         elif self.arch == 'alexnet':
    #             feat_map_s1 = self.base_s1.get_features(img_s1)
    #             heat_map = self.featmap_norm(feat_map_s1)
    #             feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
    #         elif self.arch == 'mixmodel':
    #             feat_map_r18 = self.base_s1.get_features(img_s1)
    #             feat_map_r18 = self.eca(feat_map_r18[0])
    #             feat_map_mit = self.base_s2(img_s1)
    #             feat_map_mit = self.eca(feat_map_mit[3])
    #             # feat_map_s = feat_map_r18[0] + feat_map_mit[3]  # +
    #             # feat_map_s = feat_map_r18 + feat_map_mit  # +
    #             # feat_map_s = torch.cat((feat_map_r18[0], feat_map_mit[3]), axis=1)  # cat
    #             feat_map_s = torch.cat((feat_map_r18, feat_map_mit), axis=1)
    #             heat_map = self.featmap_norm(feat_map_s)
    #             feat_map_s1 = self.pooling(feat_map_s).view(img_s1.size(0), -1)
    #         elif self.arch == 'absvit':
    #             feat_map_abs, _, _ = self.base_s1.forward_features(img_s1)
    #             feat_map_s1 = feat_map_abs[:, 1:197, :]
    #             feat_map_s1 = feat_map_s1.reshape(feat_map_s1.size(0), int(math.sqrt(feat_map_s1.size(1))), int(math.sqrt(feat_map_s1.size(1))), feat_map_s1.size(2)).permute(0, 3, 1, 2)
    #             heat_map = self.featmap_norm(feat_map_s1)
    #             feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
    #         elif self.arch == 'vit_ae':
    #             feat_map_s1 = self.base_s1.forward_features(img_s1)
    #             # logits_s1 = self.fc_s1(feat_map_s1)
    #         # logits = self.base_s1(img_s1) # absvit
    #         logits_s1 = self.fc_s1(feat_map_s1)
    #         logits = logits_s1
    #     elif self.mode == 's2':
    #         # with torch.no_grad():
    #         #     if self.arch == 'mit':
    #         #         feat_map_s1 = self.base_s1(img_s1)
    #         #         logits_s1 = self.fc_s1(self.pooling(feat_map_s1[3]).view(img_s1.size(0), -1))
    #         #         heat_map = self.featmap_norm(feat_map_s1[3])
    #         #         h4 = self.featmap_norm(feat_map_s1[3])
    #         #         h3 = self.featmap_norm(feat_map_s1[2])
    #         #         h2 = self.featmap_norm(feat_map_s1[1])
    #         #         h1 = self.featmap_norm(feat_map_s1[0])
    #         #         h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
    #         #         h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
    #         #         h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
    #         #         h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
    #         #         h_str = np.min([h_str4, h_str3, h_str2, h_str1], axis=0)
    #         #         h_end = np.max([h_end4, h_end3, h_end2, h_end1], axis=0)
    #         #         w_str = np.min([w_str4, w_str3, w_str2, w_str1], axis=0)
    #         #         w_end = np.max([w_end4, w_end3, w_end2, w_end1], axis=0)
    #         #         img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
    #         #         img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
    #         #         img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    #         #         img_gp = img_gp.view(img_s2.size(0), -1)
    #         #         feat_gp = self.linear(img_gp)
    #         #         # feat_map_s2 = self.base_s2(img_l)
    #         #         # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[3]).view(img_b, -1) + feat_gp)
    #         #     if self.arch == 'resnet18':
    #         #         feat_map_s1 = self.base_s1.get_features(img_s1)
    #         #         logits_s1 = self.fc_s1(self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1))
    #         #         heat_map = self.featmap_norm(feat_map_s1[0])
    #         #         h5 = self.featmap_norm(feat_map_s1[4])
    #         #         h4 = self.featmap_norm(feat_map_s1[3])
    #         #         h3 = self.featmap_norm(feat_map_s1[2])
    #         #         h2 = self.featmap_norm(feat_map_s1[1])
    #         #         h1 = self.featmap_norm(feat_map_s1[0])
    #         #         h_str5, h_end5, w_str5, w_end5 = self.bounding_box(h5, is_training)
    #         #         h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
    #         #         h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
    #         #         h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
    #         #         h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
    #         #         h_str = np.min([h_str5, h_str4, h_str3, h_str2, h_str1], axis=0)
    #         #         h_end = np.max([h_end5, h_end4, h_end3, h_end2, h_end1], axis=0)
    #         #         w_str = np.min([w_str5, w_str4, w_str3, w_str2, w_str1], axis=0)
    #         #         w_end = np.max([w_end5, w_end4, w_end3, w_end2, w_end1], axis=0)
    #         #         img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
    #         #         img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
    #         #         img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    #         #         img_gp = img_gp.view(img_s2.size(0), -1)
    #         #         feat_gp = self.linear(img_gp)
    #         #         # feat_map_s2 = self.base_s2.get_features(img_l)
    #         #         # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
    #         #     if self.arch == 'resnet50':
    #         #
    #         #     if self.arch == 'googlenet':
    #         #         feat_map_s1 = self.base_s1.get_features(img_s1)
    #         #         logits_s1 = self.fc_s1(self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1))
    #         #         heat_map = self.featmap_norm(feat_map_s1[0])
    #         #         h5 = self.featmap_norm(feat_map_s1[4])
    #         #         h4 = self.featmap_norm(feat_map_s1[3])
    #         #         h3 = self.featmap_norm(feat_map_s1[2])
    #         #         h2 = self.featmap_norm(feat_map_s1[1])
    #         #         h1 = self.featmap_norm(feat_map_s1[0])
    #         #         h_str5, h_end5, w_str5, w_end5 = self.bounding_box(h5, is_training)
    #         #         h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
    #         #         h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
    #         #         h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
    #         #         h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
    #         #         h_str = np.min([h_str5, h_str4, h_str3, h_str2, h_str1], axis=0)
    #         #         h_end = np.max([h_end5, h_end4, h_end3, h_end2, h_end1], axis=0)
    #         #         w_str = np.min([w_str5, w_str4, w_str3, w_str2, w_str1], axis=0)
    #         #         w_end = np.max([w_end5, w_end4, w_end3, w_end2, w_end1], axis=0)
    #         #         img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
    #         #         img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
    #         #         img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    #         #         img_gp = img_gp.view(img_s2.size(0), -1)
    #         #         feat_gp = self.linear(img_gp)
    #         #         # feat_map_s2 = self.base_s2.get_features(img_l)
    #         #         # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
    #         # if self.arch == 'mit':
    #         #     feat_map_s2 = self.base_s2(img_l)
    #         #     logits_s2 = self.fc_s2(self.pooling(feat_map_s2[3]).view(img_b, -1) + feat_gp)
    #         # if self.arch == 'resnet18':
    #         #     feat_map_s2 = self.base_s2.get_features(img_l)
    #         #     heat_map_l= self.featmap_norm(feat_map_s2[0])
    #         #     logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp) #+ feat_gp
    #         if self.arch == 'resnet50':
    #             feat_map_s1 = self.base_s1.get_features(img_s1)
    #             logits_s1 = self.fc_s1(self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1))
    #             heat_map = self.featmap_norm1(feat_map_s1[0])
    #             img_gp = heat_map * img_s2
    #             img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    #             img_gp = img_gp.view(img_s2.size(0), -1)
    #             feat_gp = self.linear(img_gp)
    #             logits_gp = self.fc_s1(feat_gp)
    #         # if self.arch == 'googlenet':
    #         #     feat_map_s2 = self.base_s2.get_features(img_l)
    #         #     logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
    #         # if is_training:
    #         #     logits = logits_s2
    #         # else:
    #         #     logits = logits_s1 + logits_s2
    #     if self.training:
    #         return logits_s1, logits_gp, feat_map_s1
    #     else:
    #         return logits_s1, logits_gp
    #     # return logits, [h_str, h_end, w_str, w_end], heat_map #@heat_map
    def forward(self, img_s1, img_s2, is_training=True):
        img_s1 = img_s1.type(dtype)  # 256*256
        img_s2 = img_s2.type(dtype)
        img_b = img_s1.size(0)
        heat_map, [h_str, h_end, w_str, w_end] = None, [None, None, None, None]
        if self.mode == 's1':
            if self.arch == 'mit':
                feat_map_s1 = self.base_s1(img_s1)
                heat_map = self.featmap_norm(feat_map_s1[3])
                feat_map_s1 = self.pooling(feat_map_s1[3]).view(img_s1.size(0), -1)
            elif self.arch == 'resnet18':
                feat_map_s1 = self.base_s1.get_features(img_s1)
                # f3 = self.c3(feat_map_s1[0])
                # f3 = self.sc3(f3)
                # f3 = self.c3(f3)
                # feat_map_s1sc = feat_map_s1[0] + f3
                heat_map = self.featmap_norm(feat_map_s1[0])
                feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
                # heat_map = self.featmap_norm(feat_map_s1[0])
                # feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
            elif self.arch == 'googlenet':
                feat_map_s1 = self.base_s1.get_features(img_s1)
                heat_map = self.featmap_norm(feat_map_s1[0])
                feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
            elif self.arch == 'resnet50':
                feat_map_s1 = self.base_s1.get_features(img_s1)
                heat_map = self.featmap_norm(feat_map_s1[0])
                feat_map_s1 = self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1)
            elif self.arch == 'vgg16':
                feat_map_s1 = self.base_s1.get_features(img_s1)
                heat_map = self.featmap_norm(feat_map_s1)
                feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
            elif self.arch == 'alexnet':
                feat_map_s1 = self.base_s1.get_features(img_s1)
                heat_map = self.featmap_norm(feat_map_s1)
                feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
            elif self.arch == 'mixmodel':
                feat_map_r18 = self.base_s1.get_features(img_s1)
                feat_map_r18 = self.eca(feat_map_r18[0])
                feat_map_mit = self.base_s2(img_s1)
                feat_map_mit = self.eca(feat_map_mit[3])
                # feat_map_s = feat_map_r18[0] + feat_map_mit[3]  # +
                # feat_map_s = feat_map_r18 + feat_map_mit  # +
                # feat_map_s = torch.cat((feat_map_r18[0], feat_map_mit[3]), axis=1)  # cat
                feat_map_s = torch.cat((feat_map_r18, feat_map_mit), axis=1)
                heat_map = self.featmap_norm(feat_map_s)
                feat_map_s1 = self.pooling(feat_map_s).view(img_s1.size(0), -1)
            elif self.arch == 'absvit':
                feat_map_abs, _, _ = self.base_s1.forward_features(img_s1)
                feat_map_s1 = feat_map_abs[:, 1:197, :]
                feat_map_s1 = feat_map_s1.reshape(feat_map_s1.size(0), int(math.sqrt(feat_map_s1.size(1))), int(math.sqrt(feat_map_s1.size(1))), feat_map_s1.size(2)).permute(0, 3, 1, 2)
                heat_map = self.featmap_norm(feat_map_s1)
                feat_map_s1 = self.pooling(feat_map_s1).view(img_s1.size(0), -1)
            elif self.arch == 'vit_ae':
                feat_map_s1 = self.base_s1.forward_features(img_s1)
                # logits_s1 = self.fc_s1(feat_map_s1)
            # logits = self.base_s1(img_s1) # absvit
            logits_s1 = self.fc_s1(feat_map_s1)
            logits = logits_s1
        elif self.mode == 's2':
            with torch.no_grad():
                if self.arch == 'mit':
                    feat_map_s1 = self.base_s1(img_s1)
                    logits_s1 = self.fc_s1(self.pooling(feat_map_s1[3]).view(img_s1.size(0), -1))
                    heat_map = self.featmap_norm(feat_map_s1[3])
                    h4 = self.featmap_norm(feat_map_s1[3])
                    h3 = self.featmap_norm(feat_map_s1[2])
                    h2 = self.featmap_norm(feat_map_s1[1])
                    h1 = self.featmap_norm(feat_map_s1[0])
                    h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
                    h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
                    h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
                    h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
                    h_str = np.min([h_str4, h_str3, h_str2, h_str1], axis=0)
                    h_end = np.max([h_end4, h_end3, h_end2, h_end1], axis=0)
                    w_str = np.min([w_str4, w_str3, w_str2, w_str1], axis=0)
                    w_end = np.max([w_end4, w_end3, w_end2, w_end1], axis=0)
                    img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
                    img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
                    img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    img_gp = img_gp.view(img_s2.size(0), -1)
                    feat_gp = self.linear(img_gp)
                    # feat_map_s2 = self.base_s2(img_l)
                    # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[3]).view(img_b, -1) + feat_gp)
                if self.arch == 'resnet18':
                    feat_map_s1 = self.base_s1.get_features(img_s1)
                    logits_s1 = self.fc_s1(self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1))
                    heat_map = self.featmap_norm(feat_map_s1[0])
                    h5 = self.featmap_norm(feat_map_s1[4])
                    h4 = self.featmap_norm(feat_map_s1[3])
                    h3 = self.featmap_norm(feat_map_s1[2])
                    h2 = self.featmap_norm(feat_map_s1[1])
                    h1 = self.featmap_norm(feat_map_s1[0])
                    h_str5, h_end5, w_str5, w_end5 = self.bounding_box(h5, is_training)
                    h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
                    h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
                    h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
                    h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
                    h_str = np.min([h_str5, h_str4, h_str3, h_str2, h_str1], axis=0)
                    h_end = np.max([h_end5, h_end4, h_end3, h_end2, h_end1], axis=0)
                    w_str = np.min([w_str5, w_str4, w_str3, w_str2, w_str1], axis=0)
                    w_end = np.max([w_end5, w_end4, w_end3, w_end2, w_end1], axis=0)
                    img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
                    img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
                    img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    img_gp = img_gp.view(img_s2.size(0), -1)
                    feat_gp = self.linear(img_gp)
                    # feat_map_s2 = self.base_s2.get_features(img_l)
                    # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
                if self.arch == 'resnet50':
                    feat_map_s1 = self.base_s1.get_features(img_s1)
                    logits_s1 = self.fc_s1(self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1))
                    heat_map = self.featmap_norm(feat_map_s1[0])
                    h5 = self.featmap_norm(feat_map_s1[4])
                    h4 = self.featmap_norm(feat_map_s1[3])
                    h3 = self.featmap_norm(feat_map_s1[2])
                    h2 = self.featmap_norm(feat_map_s1[1])
                    h1 = self.featmap_norm(feat_map_s1[0])
                    h_str5, h_end5, w_str5, w_end5 = self.bounding_box(h5, is_training)
                    h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
                    h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
                    h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
                    h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
                    h_str = np.min([h_str5, h_str4, h_str3, h_str2, h_str1], axis=0)
                    h_end = np.max([h_end5, h_end4, h_end3, h_end2, h_end1], axis=0)
                    w_str = np.min([w_str5, w_str4, w_str3, w_str2, w_str1], axis=0)
                    w_end = np.max([w_end5, w_end4, w_end3, w_end2, w_end1], axis=0)
                    img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
                    img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
                    img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    img_gp = img_gp.view(img_s2.size(0), -1)
                    feat_gp = self.linear(img_gp)
                    # feat_map_s2 = self.base_s2.get_features(img_l)
                    # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
                if self.arch == 'googlenet':
                    feat_map_s1 = self.base_s1.get_features(img_s1)
                    logits_s1 = self.fc_s1(self.pooling(feat_map_s1[0]).view(img_s1.size(0), -1))
                    heat_map = self.featmap_norm(feat_map_s1[0])
                    h5 = self.featmap_norm(feat_map_s1[4])
                    h4 = self.featmap_norm(feat_map_s1[3])
                    h3 = self.featmap_norm(feat_map_s1[2])
                    h2 = self.featmap_norm(feat_map_s1[1])
                    h1 = self.featmap_norm(feat_map_s1[0])
                    h_str5, h_end5, w_str5, w_end5 = self.bounding_box(h5, is_training)
                    h_str4, h_end4, w_str4, w_end4 = self.bounding_box(h4, is_training)
                    h_str3, h_end3, w_str3, w_end3 = self.bounding_box(h3, is_training)
                    h_str2, h_end2, w_str2, w_end2 = self.bounding_box(h2, is_training)
                    h_str1, h_end1, w_str1, w_end1 = self.bounding_box(h1, is_training)
                    h_str = np.min([h_str5, h_str4, h_str3, h_str2, h_str1], axis=0)
                    h_end = np.max([h_end5, h_end4, h_end3, h_end2, h_end1], axis=0)
                    w_str = np.min([w_str5, w_str4, w_str3, w_str2, w_str1], axis=0)
                    w_end = np.max([w_end5, w_end4, w_end3, w_end2, w_end1], axis=0)
                    img_gp = self.img_sampling(img_s2, h_str, h_end, w_str, w_end)
                    img_l = self.img_sampling(img_s1, h_str, h_end, w_str, w_end)
                    img_gp = img_gp.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    img_gp = img_gp.view(img_s2.size(0), -1)
                    feat_gp = self.linear(img_gp)
                    # feat_map_s2 = self.base_s2.get_features(img_l)
                    # logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
            if self.arch == 'mit':
                feat_map_s2 = self.base_s2(img_l)
                logits_s2 = self.fc_s2(self.pooling(feat_map_s2[3]).view(img_b, -1) + feat_gp)
            if self.arch == 'resnet18':
                feat_map_s2 = self.base_s2.get_features(img_l)
                heat_map_l= self.featmap_norm(feat_map_s2[0])
                logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp) #+ feat_gp
            if self.arch == 'resnet50':
                feat_map_s2 = self.base_s2.get_features(img_l)
                heat_map_l = self.featmap_norm(feat_map_s2[0])
                logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)  # + feat_gp
            if self.arch == 'googlenet':
                feat_map_s2 = self.base_s2.get_features(img_l)
                logits_s2 = self.fc_s2(self.pooling(feat_map_s2[0]).view(img_b, -1) + feat_gp)
            if is_training:
                logits = logits_s2
            else:
                logits = logits_s1 + logits_s2
        return logits, [h_str, h_end, w_str, w_end], heat_map #@heat_map