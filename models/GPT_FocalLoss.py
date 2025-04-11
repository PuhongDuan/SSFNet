import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        self.weight = self.weight.to(inputs.device)
        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets_onehot)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, weight=None, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction
# 
#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
# 
#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
# 
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
# #
# import torch
# import torch.nn.functional as F
#
# class FocalLoss(torch.nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.Tensor(alpha)
#         self.gamma = gamma
#
#     def forward(self, logits, labels):
#         self.alpha = self.alpha.to(logits.device)
#         logits = logits * self.alpha
#         ce_loss = F.cross_entropy(logits, labels, reduction='none')
#         pt = torch.exp(-ce_loss)
#         # focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss.mean()
#         focal_loss = (1 - pt) ** self.gamma * ce_loss.mean()
#         return focal_loss


