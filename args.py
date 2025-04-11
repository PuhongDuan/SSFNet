import argparse

def args_parser():
	parser = argparse.ArgumentParser(description='Build the splits of remote datasets')
	
	# root setting
	# parser.add_argument('--data_dir', default='/home/huangwei/Datasets/Remote_Sensing_Scene_Classification', type=str)
	parser.add_argument('--data_dir', default='/mnt/home/zhengjialin/SKAL/datasets', type=str)
	parser.add_argument('--dataset', default='UCM', type=str, choices=['AID', 'UCM', 'RGB_data', 'WHU', 'NWPU-RESISC45', 'rgb_gp', 'H_data', 'rgb_gp_zs', 'CQU_data'])
	parser.add_argument('--class_list', default='dataset/splits/classInd.txt', type=str)
	parser.add_argument('--train_list', default='dataset/splits/train_split.txt', type=str)
	parser.add_argument('--val_list', default='dataset/splits/val_split.txt', type=str)
	parser.add_argument('--read_path', default='checkpoints/dataset_arch.pth', type=str)
	parser.add_argument('--save_path', default='checkpoints/dataset_arch.pth', type=str)
	parser.add_argument('--work_dir', default='checkpoints', type=str)
	parser.add_argument('--traget_path', default='./image.mat', type=str)
	parser.add_argument('--traget_label', default=1, type=int)
	parser.add_argument('--patch_size', default=224, type=int)

	# model alexnet
	parser.add_argument('--arch', default='alexnet', type=str, choices=['HSINet', 'mixmodel', 'mit', 'resnet50', 'vit_ae','absvit','alexnet', 'googlenet', 'resnet18', 'vgg16'])
	parser.add_argument('--mode', default='s1', type=str, choices=['s1', 's2', 's3', 's51','s52','skal', 's5','fz', 'fz_1', 'gp', 'a'])
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--img_size', default=224, type=int)
	parser.add_argument('--energy_thr', default=0.7, type=float)	
	parser.add_argument('--n_workers', default=8, type=int)

	# train setting
	parser.add_argument('--start_epoch', default=0, type=int)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--step_size', default=20, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)  #一般为1e-4

	parser.add_argument('--data_path', type=str, help="数据集所在路径", default='./datasets/')
	parser.add_argument('--model', type=str, choices=['vgg16'], default='vgg16')
	parser.add_argument('--K', type=int, help="降为参数", default=784)
	parser.add_argument('--ratio', type=float, help="训练比例", default=0.2)
	parser.add_argument('--extract', action='store_true', help="是否进行特征提取")
	parser.add_argument('--train', action='store_true', help="是否进行特征训练")
	args = parser.parse_args()

	return args

