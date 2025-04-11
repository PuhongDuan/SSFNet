import torch
from torch.autograd import Variable
from utils import *
import csv
import os
from pycm import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, train_loader, net, optim, criterion):
    print('train at epoch {}'.format(epoch))
    net.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    torch.autograd.set_detect_anomaly(True)
    
    for i, (imgs_s1, imgs_s2, labels) in enumerate(train_loader):
        img_s1 = Variable(imgs_s1.cuda())
        img_s2 = Variable(imgs_s2.cuda())
        labels = Variable(labels.cuda())
        # img_s1 = imgs_s1.cuda()
        # img_s2 = imgs_s2.cuda()
        # labels = labels.cuda()
        logits, _, _ = net(img_s1, img_s2, is_training=True)
        optim.zero_grad()
        loss = criterion(logits, labels)
        # loss.requires_grad_(True)
        # loss.mean().backward()
        loss.backward()
        optim.step()

        acc = accuracy(logits, labels)
        losses.update(loss.item(), logits.size(0))
        # losses.update(loss.mean().item(), logits.size(0))
        accuracies.update(acc, logits.size(0))

        if (i%5==0 and i!=0) or i+1==len(train_loader):
            print ('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   Accu:{:.2f}%'.\
                    format(epoch, i, len(train_loader), float(losses.avg), float(accuracies.avg)*100))
       
    return accuracies.avg


def train_mcc(epoch, train_iter, val_iter, net, optim, criterion, mcc_loss):
    print('train at epoch {}'.format(epoch))
    net.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    losses_s = AverageMeter()
    accuracies_s = AverageMeter()
    losses_gp = AverageMeter()
    accuracies_gp = AverageMeter()
    torch.autograd.set_detect_anomaly(True)

    for i in range(len(train_iter)):
        x_s, labels_s = next(train_iter)[:2]
        x_t, = next(val_iter)[:1]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        ####
        # y, f = net(x, is_training=True)
        # y_s, y_t = y.chunk(2, dim=0)
        # cls_loss = criterion(y_s, labels_s)
        # transfer_loss = mcc_loss(y_t)
        # loss = cls_loss + transfer_loss
        # 
        # optim.zero_grad()
        # loss.backward()
        # optim.step()
        # 
        # acc = accuracy(y_s, labels_s)
        # losses.update(loss.item(), y_s.size(0))
        # accuracies.update(acc, y_s.size(0))
        ####
        y, y_gp, f = net(x, is_training=True)
        y_s, y_t = y.chunk(2, dim=0)
        ygp_s, ygp_t = y_gp.chunk(2, dim=0)
        cls_loss = criterion((y_s + ygp_s)/2, labels_s)
        loss_s = criterion(y_s, labels_s)
        loss_gp = criterion(ygp_s, labels_s)
        transfer_loss = mcc_loss((y_t + ygp_t)/2)
        # r1 = loss2 / loss, r2 = loss1 / loss,
        loss = cls_loss + transfer_loss
        # loss.requires_grad_(True)
        # loss.mean().backward()
        optim.zero_grad()
        loss.backward()
        optim.step()

        acc_s = accuracy(y_s, labels_s)
        acc_gp = accuracy(ygp_s, labels_s)
        acc = accuracy((y_s + ygp_s)/2, labels_s)

        losses.update(loss.item(), y_s.size(0))
        losses_s.update(loss_s.item(), y_s.size(0))
        losses_gp.update(loss_gp.item(), y_s.size(0))
        accuracies.update(acc, y_s.size(0))
        accuracies_s.update(acc_s, y_s.size(0))
        accuracies_gp.update(acc_gp, y_s.size(0))

        if (i % 50 == 0 and i != 0) or i + 1 == len(train_iter):
            print('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   Accu:{:.2f}%    Loss_s:{:.4f}   Accu_s:{:.2f}%  Loss_gp:{:.4f}   Accu_gp:{:.2f}%'. \
                  format(epoch, i, len(train_iter), float(losses.avg), float(accuracies.avg) * 100, float(losses_s.avg), float(accuracies_s.avg) * 100, float(losses_gp.avg), float(accuracies_gp.avg) * 100))
        
        # if (i % 50 == 0 and i != 0) or i + 1 == len(train_loader):
        #     print('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   Accu:{:.2f}%'. \
        #           format(epoch, i, len(train_loader), float(losses.avg), float(accuracies.avg) * 100))
            
    return accuracies.avg

def train1(epoch, train_loader, net, optim, criterion):
    print('train at epoch {}'.format(epoch))

    net.train()
    losses = AverageMeter()
    accuracies = AverageMeter()

    for i, (imgs_s1, imgs_s2, labels) in enumerate(train_loader):
        img_s1 = Variable(imgs_s1.cuda())
        img_s2 = Variable(imgs_s2.cuda())
        labels = Variable(labels.cuda())
        img_s1 = img_s1[:, :, 16: 240, 16: 240]
        logits_t = net(img_s1)
        logits = (logits_t[0]+logits_t[0])/2
        optim.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()

        acc = accuracy(logits, labels)
        losses.update(loss.item(), logits.size(0))
        accuracies.update(acc, logits.size(0))

        if (i % 50 == 0 and i != 0) or i + 1 == len(train_loader):
            print('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   Accu:{:.2f}%'. \
                  format(epoch, i, len(train_loader), float(losses.avg), float(accuracies.avg) * 100))

    return accuracies.avg

