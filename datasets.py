import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import random
import scipy.io as scio
import numpy as np
import hdf5storage

class MultiScaleRandomCrop(object):
    def __init__(self, scales, size):
        self.scales = scales
        self.crop_size = size

    def __call__(self, img):
        img_size = img.size[0] 
        scale = random.sample(self.scales, 1)[0]
        re_size = int(img_size / scale)
        img = img.resize((re_size, re_size), Image.BILINEAR)
        x1 = random.randint(0, re_size-img_size)
        y1 = random.randint(0, re_size-img_size)
        x2 = x1 + self.crop_size
        y2 = y1 + self.crop_size
        img = img.crop((x1, y1, x2, y2))
        return img


def make_list(root, split_path):
	list_path = os.path.join(root, split_path)
	data_list = []
	class_dict = {}
	f = open(list_path, 'r')
	line = f.readline()
	while line:
		sample ={}
		line = line.strip('\n')
		img_path, label = line.split('  ')

		sample['img_path'] = img_path
		sample['label'] = label
		data_list.append(sample)
		if label not in class_dict.keys(): 	
			class_dict[label] = [img_path]
		else:
			class_dict[label].append(img_path)

		line = f.readline()
	f.close()
	return data_list, class_dict
def zsjs(h):
	R = h[:,:,29]
	G = h[:,:,20]
	B = h[:,:,12]
	NIR = h[:,:,46]
	MIR = h[:,:,134]
	SWIR = h[:,:,196]
	NDVI = (NIR - R) / (NIR + R)
	NBR = (NIR - SWIR) / (NIR + SWIR)
	MNDWI = (G - MIR) / (G + MIR)
	NDBI = (MIR - NIR) / (MIR + NIR)
	SI = ((SWIR + R) - (NIR + B)) / ((SWIR + R) + (NIR + B))
	def scale_2d(data):
		M = data.max()
		m = data.min()
		data = (data - m) / (M - m)
		return data

	NDVI = scale_2d(NDVI)
	MNDWI = scale_2d(MNDWI)
	NDBI = scale_2d(NDBI)
	SI = scale_2d(SI)
	# zs = np.zeros((5, 256, 256))
	# zs[0, :, :] = NDVI
	# zs[1, :, :] = NBR
	# zs[2, :, :] = MNDWI
	# zs[3, :, :] = NDBI
	# zs[4, :, :] = SI
	zs = np.zeros((256, 256, 3))
	zs[:, :, 0] = NDVI
	zs[:, :, 1] = MNDWI
	zs[:, :, 2] = SI

	return zs

# def zhishujisuan(self, img_s2):
# 	a, b, c, d = img_s2.size
# 	img = np.zeros([a,11])
# 	for i in range(a):
# 		data = img_s2[i, :, :, :]
# 		water = data[3, :, :]
# 		if (water > 0.8):
#
# 		index = find(water == 1);
# 		zs_w(i, 1) = size(index, 1) / (256 * 256);
# 	return img


class datasets(data.Dataset):
	def __init__(self, root, split_path, transform_s1, transform_s2):
		self.root = root
		self.split_path = split_path 
		self.data_list, self.class_dict = make_list(root, split_path)

		self.transform_s1 = transform_s1
		self.transform_s2 = transform_s2
	# def __getitem__(self, idx):
	# 	label = int(self.data_list[idx]['label'])
	# 	img_pth = self.data_list[idx]['img_path']
	# 	img_pth = os.path.join(self.root, img_pth)
	# 	# img = Image.open(img_pth).convert('RGB')
	# 	data = scio.loadmat(img_pth)  # 读取两个数据,rgb和gp的
	# 	data = data['data']
	# 	img_1 = data[:, :, [30, 20, 12]]
	# 	img_2 = data
	# 	# img_2 = np.transpose(img_2, (1, 0))
	# 	img_1 = np.transpose(img_1, (2, 0, 1))
	# 	img_2 = np.transpose(img_2, (2, 0, 1))
	# 	img_s1 = img_1 # self.transform_s1(img_1)
	# 	img_s2 = img_2 # self.transform_s2(img) #img #
	# 	return img_s1, img_s2, label
	# def __getitem__(self, idx):
	# 	label = int(self.data_list[idx]['label'])
	# 	img_pth = self.data_list[idx]['img_path']
	# 	img_pth = os.path.join(self.root, img_pth)
	# 	# img = Image.open(img_pth).convert('RGB')
	# 	data = scio.loadmat(img_pth)  # 读取两个数据,rgb和gp的
	# 	img = data['data']
	# 	img_1 = img[:,:,[30,20,12]]
	# 	# zs = zsjs(img_1)
	# 	img_2 = img # [:, :, [1, 3, 4]]
	# 	img_2 = np.transpose(img_2, (2, 0, 1))
	# 	img_1 = np.transpose(img_1, (2, 0, 1))
	# 	# img_1 = img_1.astype('float32')
	# 	# water = img_2[3, :, :]
	# 	# global w
	# 	# w = np.zeros(11)# water[water >= 0.8] = 1
	# 	# wt = np.where(water >= 0.8, 1, 0)  # 把a1中所有小于10的数全部变成1，其余的变成0
	# 	# s = np.sum(sum(wt))/(256*256)
	# 	# if s >= 0.5:
	# 	# 	# w[3], w[5], w[7: 11] = 0, 0, 0
	# 	# 	w[0:3], w[4], w[6] = -0.99, -0.99, -0.99
	# 	# 	# w = np.where(w == 0, -0.99, 1)
	# 	# else:
	# 	# 	w = w
	# 	img_s1 = np.nan_to_num(img_1) # self.transform_s1(img_1)
	# 	img_s2 = np.nan_to_num(img_2) # self.transform_s2(img) #img #
	# 	return img_s1, img_s2, label #
	def __getitem__(self, idx):
		label = int(self.data_list[idx]['label'])
		img_pth = self.data_list[idx]['img_path']
		img_pth = os.path.join(self.root, img_pth)
		# img = Image.open(img_pth).convert('RGB')
		# data = scio.loadmat(img_pth)  # 读取两个数据,rgb和gp的
		data = hdf5storage.loadmat(img_pth)
		img = data['data']
		img_1 = img[:,:,[19,13,7]]
		# zs = zsjs(img)
		img_2 = img # [:, :, [1, 3, 4]]
		img_2 = np.transpose(img_2, (2, 0, 1))
		img_1 = np.transpose(img_1, (2, 0, 1))
		# img_1 = img_1.astype('float32')
		img_s1 = np.nan_to_num(img_1) # self.transform_s1(img_1)
		img_s2 = np.nan_to_num(img_2) # self.transform_s2(img) #img #
		return img_s1, img_s2, label #
	def __len__(self):
		return len(self.data_list)


def load_datasets(root, train_list, val_list, mode, batch_size, img_size, n_workers):
	normalize = transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
		)

	train_transform = transforms.Compose([
		transforms.Resize(int(img_size)),
		transforms.CenterCrop(img_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
		])
	val_transform = transforms.Compose([
		transforms.Resize(img_size),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		normalize,
		])
	transform_s2 = transforms.Compose([
		transforms.Resize(int(img_size*2)),
		transforms.CenterCrop(int(img_size*2)),
		transforms.ToTensor(),
		normalize,
		])

	train_datasets = datasets(root=root, 
							  split_path=train_list, 
							  transform_s1=train_transform, 
							  transform_s2=transform_s2)
	val_datasets = datasets(root=root, 
							split_path=val_list, 
  							transform_s1=val_transform,
							transform_s2=transform_s2)
	train_loader = torch.utils.data.DataLoader(
							dataset=train_datasets,
							batch_size=batch_size,
							shuffle=True,
							num_workers=n_workers,
							pin_memory=True,
							drop_last=True)
	val_loader = torch.utils.data.DataLoader(
							dataset=val_datasets,
							batch_size=batch_size,
							shuffle=True,
							num_workers=n_workers,
							pin_memory=True, drop_last=True)
	test_loader = torch.utils.data.DataLoader(
							dataset=val_datasets,
							batch_size=1,
							shuffle=True,
							num_workers=n_workers,
							pin_memory=True, drop_last=True)
	return train_loader, val_loader #, test_loader

