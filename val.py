import torch
from torch.autograd import Variable
from utils import *
from pycm import *
import numpy as np
import torch.nn.functional as F
import os

def validation_mcc(epoch, best_acc, val_loader, net, resume_path, criterion):
	print('val at epoch {}'.format(epoch))
	net.eval()
	losses = AverageMeter()
	accuracies = AverageMeter()

	with torch.no_grad():
		for i, (imgs_s, labels) in enumerate(val_loader):
			with torch.no_grad():
				imgs_s = Variable(imgs_s.cuda())
				labels = Variable(labels.cuda())
				logits_s, logits_gp = net(imgs_s, is_training=False)
				loss_s = criterion(logits_s, labels)
				loss_gp = criterion(logits_gp, labels)
				loss = loss_s + loss_gp
				r1 = loss_gp / loss
				r2 = loss_s / loss
				acc = accuracy(logits_s*r1+loss_gp*r2, labels)
			# losses.update(loss.mean().item(), logits.size(0))
			losses.update(loss.item(), logits_s.size(0))
			accuracies.update(acc, logits_s.size(0))

			if (i%5==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'.   \
						format(epoch, i, len(val_loader), float(losses.avg), float(accuracies.avg)*100))

	print ('best_acc: {:.2f}%'.format(best_acc*100))
	print ('curr_acc: {:.2f}%'.format(accuracies.avg*100))
	if accuracies.avg >= best_acc:
		best_acc = accuracies.avg
		save_file_path = resume_path
		states = {'state_dict': net.state_dict(),
				  'epoch':epoch,
				  'acc':best_acc}
		torch.save(states, save_file_path)
		print ('Saved!')
	print ('')
	return best_acc, accuracies.avg

def validation(epoch, best_acc, val_loader, net, resume_path, criterion):
	print('val at epoch {}'.format(epoch))
	net.eval()
	losses = AverageMeter()
	accuracies = AverageMeter()

	with torch.no_grad():
		for i, (imgs_s1, imgs_s2, labels) in enumerate(val_loader):
			with torch.no_grad():
				img_s1 = Variable(imgs_s1.cuda())
				img_s2 = Variable(imgs_s2.cuda())
				labels = Variable(labels.cuda())
				# a = os.environ["CUDA_VISIBLE_DEVICES"]
				# print(a)
				# img_s1 = imgs_s1.cuda()
				# img_s2 = imgs_s2.cuda()
				# labels = labels.cuda()
				logits, _, _ = net(img_s1, img_s2, is_training=False)
				loss = criterion(logits, labels)
				acc = accuracy(logits, labels)
			# losses.update(loss.mean().item(), logits.size(0))
			losses.update(loss.item(), logits.size(0))
			accuracies.update(acc, logits.size(0))

			if (i%5==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'.   \
						format(epoch, i, len(val_loader), float(losses.avg), float(accuracies.avg)*100))

	print ('best_acc: {:.2f}%'.format(best_acc*100))
	print ('curr_acc: {:.2f}%'.format(accuracies.avg*100))
	if accuracies.avg >= best_acc:
		best_acc = accuracies.avg
		save_file_path = resume_path
		states = {'state_dict': net.state_dict(),
				  'epoch':epoch,
				  'acc':best_acc}
		torch.save(states, save_file_path)
		print ('Saved!')
	print ('')		
	return best_acc, accuracies.avg

def validation1(epoch, best_acc, val_loader, net, resume_path, criterion):
	print('val at epoch {}'.format(epoch))
	net.eval()
	losses = AverageMeter()
	accuracies = AverageMeter()

	with torch.no_grad():
		for i, (imgs_s1, imgs_s2, labels) in enumerate(val_loader):
			with torch.no_grad():
				img_s1 = Variable(imgs_s1.cuda())
				img_s2 = Variable(imgs_s2.cuda())
				labels = Variable(labels.cuda())
				img_s1 = img_s1[:, :, 16: 240, 16: 240]
				logits = net(img_s1)
				loss = criterion(logits, labels)
				acc = accuracy(logits, labels)
			losses.update(loss.item(), logits.size(0))
			accuracies.update(acc, logits.size(0))

			if (i%50==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'.   \
						format(epoch, i, len(val_loader), float(losses.avg), float(accuracies.avg)*100))

	print ('best_acc: {:.2f}%'.format(best_acc*100))
	print ('curr_acc: {:.2f}%'.format(accuracies.avg*100))
	if accuracies.avg >= best_acc:
		best_acc = accuracies.avg
		save_file_path = resume_path
		states = {'state_dict': net.state_dict(),
				  'epoch':epoch,
				  'acc':best_acc}
		torch.save(states, save_file_path)
		print ('Saved!')
	print ('')
	return best_acc, accuracies.avg