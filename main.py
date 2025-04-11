import os
import torch
import torch.nn as nn
from args import args_parser
from train import train
from val import validation
from datasets import load_datasets
from collections import OrderedDict
from models.full_model import *
import pdb
from models.GPT_FocalLoss import *
from time import time
# from utils import *
# 固定随机种子
def set_seed(seed=0):
    # print('seed = {}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 使用lstm需要添加下述环境变量为:16:8，如果cuda版本为10.2，去百度一下应该将环境变量设为多少。
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = True
    # use_deterministic_algorithms用于自查自己的代码是否包含不确定的算法，报错说明有，根据报错位置查询并替代该处的算法。1.8之前的版本好像没有此方法。
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(seed=0)
# torch.cuda.set_device(7)
os.environ["CUDA_VISIBLE_DEVICES"]= '0'  #cudaxinghao
args = args_parser()
best_acc = 0

if __name__ == '__main__':	
    # load datasets feat
    train_list = args.train_list.replace('dataset', args.dataset)
    val_list = args.val_list.replace('dataset', args.dataset)
    train_loader, val_loader = load_datasets(args.data_dir, 
                                             train_list, 
                                             val_list,
                                             args.mode,  
                                             args.batch_size, 
                                             args.img_size, 
                                             args.n_workers)

    # bulid model
    # resume_path = args.resume_path.replace('dataset', args.dataset)  \
    #                               .replace('arch', args.arch)
    if args.dataset=='AID':
        n_classes = 30
    elif args.dataset=='UCM':
        n_classes = 21
    elif args.dataset=='RGB_data':
        n_classes = 11
    elif args.dataset=='rgb_gp':
        n_classes = 11
    elif args.dataset == 'H_data':
        n_classes = 11
    elif args.dataset == 'rgb_gp_zs':
        n_classes = 11
    elif args.dataset == 'CQU_data':
        n_classes = 5


    net = FullModel(arch=args.arch,
                    n_classes=n_classes,
                    mode=args.mode,
                    energy_thr=args.energy_thr,
                    ).cuda()

    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    if os.path.exists(args.read_path):
        resume = torch.load(args.read_path)
        net.load_state_dict(resume['state_dict'], strict=False)
        print('Load checkpoint {}'.format(args.read_path))

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    sche = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size)


    all_time = 0
    best_acc, val_acc = validation(0, best_acc, val_loader, net, args.save_path, criterion)
    # pdb.set_trace()
    file_name = './{}/{}_{}.txt'.format(args.work_dir, args.dataset, args.mode)
    with open(file_name, 'a') as file:
        file.write(args.save_path + '\n')

    for i in range(args.start_epoch, args.epochs):
        beg_time = time()
        train_acc = train(i, train_loader, net, optim, criterion)
        end_time = time()
        all_time = all_time + (end_time - beg_time)
        print('training_time: ', all_time)

        str_time = time()
        best_acc, val_acc = validation(i, best_acc, val_loader, net, args.save_path, criterion)
        end_time = time()
        valid_time = end_time - str_time
        print('valid_time: ', valid_time)

        with open(file_name, 'a') as file:
            file.write(str(i) + ' ' + str(val_acc) + ' ' + str(best_acc) + ' ' + '\n')

        sche.step()