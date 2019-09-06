import os
# Using this code to force the usage of any specific GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torch.utils.data as data
import time
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable
from math import log10
import torchvision
import cv2
import skimage
import scipy.io
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from model import loss
from utils.model_storage import save_checkpoint
from data.dataset import *
from model.FSRCNN import *
from model import FALSR_A
from model import FALSR_B
from model import SRCNN
from model import VDSR
from model.ESPCN import *



parser = argparse.ArgumentParser()
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--lr", default="2.5e-4", type=float, help="The learning rate of our network")
parser.add_argument("--save_freq", default="2", type=float, help="The intervals of our model storage intervals")
parser.add_argument("--iter_freq", default="5", type=float, help="The intervals of our model's evaluation intervals")
parser.add_argument("--result_dir", default="./result/ESPCN/", type=str, help="The path of our result images")
parser.add_argument("--model_path", default="./weights/", type=str, help="The path to store our model")
parser.add_argument("--epochs", default="1000", type=int, help="The path to∂ store our model")
parser.add_argument("--start_epoch", default="0", type=int, help="The path to store our model")
parser.add_argument("--batch_size", default="16", type=int, help="The path to store our batch_size")
parser.add_argument("--model",default = "ESPCN",type = str, help = "The model to train")
parser.add_argument("--upscale",default= "4",type = int,help = "The upscale factor")
parser.add_argument("--criterion",default = "l1",type = str,help = "The loss function design")

global opt,model
opt = parser.parse_args()
start_time = time.time()

demo_dataset_x2 = ImageDatasetFromFile("/media/lab216/Storage/dataset/Wangxintao数据集/DIV2K800/DIV2K_train_HR/DIV2K_train_HR/",upscale=2.0)
demo_dataset_x4 = ImageDatasetFromFile("/media/lab216/Storage/dataset/Wangxintao数据集/DIV2K800/DIV2K_train_HR/DIV2K_train_HR/",upscale=4.0)

demo_dataset_x2_scale = ImageDatasetFromFile("./DIV2K800/train/DIV2K_train_HR/DIV2K_train_HR/",upscale=2.0,is_scale_back=True)
demo_dataset_x4_scale = ImageDatasetFromFile("./DIV2K800/train/DIV2K_train_HR/DIV2K_train_HR/",upscale=4.0,is_scale_back=True)

train_data_loader = data.DataLoader(dataset=demo_dataset_x4, batch_size=opt.batch_size, num_workers=8,drop_last=True,pin_memory=True)


if opt.model:
    if opt.model == "FSRCNN" and opt.upscale == 4:
        model = FSRCNN(num_channels = 3,upscale_factor=4)

    if opt.model == "FSRCNN" and opt.upscale == 4:
        model = FSRCNN(num_channels = 3,upscale_factor=4)

    if opt.model == "FALSR_A" or opt.model == "FALSR_B":
        if opt.upscale is not 2:
            raise ("ONLY SUPPORT 2X")
        else:
            if opt.model == "FALSR_A":
                model = FALSR_A()
            if opt.model == "FALSR_B":
                model = FALSR_B()

    if opt.model == "SRCNN" and opt.upscale == 4:
        model = SRCNN(num_channels=3,upscale_factor=4)

    if opt.model == "VDSR" and opt.upscale == 4:
        model = VDSR(num_channels=3,base_channels=3,num_residual=20)

    if opt.model == "ESPCN" and opt.upscale == 4:
        model = ESPCN(num_channels = 3,feature = 64, upscale_factor = 4)

if opt.criterion:
    if opt.criterion == "l1":
        criterion = nn.L1Loss()
    if opt.criterion == "l2":
        criterion = nn.MSELoss()
    if opt.criterion == "custom":
        pass

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optimizerG = optim.RMSprop(model.parameters(),lr = opt.lr)

if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)

        # debug
        pretrained_dict = weights['model'].state_dict()
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

for epoch in range(opt.start_epoch,opt.epochs):
    if epoch % opt.save_freq == 0:
        #model, epoch, model_path, iteration, prefix=""
        save_checkpoint(model, epoch, opt.model_path,0, prefix= opt.model)

    for iteration, batch in enumerate(train_data_loader):
        lr, hr = Variable(batch[0]),Variable(batch[1])
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()

        sr = model(lr)
        model.zero_grad()
        loss = criterion(sr,hr)
        loss.backward()
        optimizerG.step()

        info = "===> Epoch[{}]({}/{}): time: {:4.4f}:\n".format(epoch, iteration, len(demo_dataset_x4) // opt.batch_size,
                                                              time.time() - start_time)

        info += "Loss: {:.4f}\n".format(loss.float())
        print(info)

        if iteration % opt.iter_freq == 0:
            # model, epoch, model_path, iteration, prefix=""
            # if not os.path.isdir(opt.result_dir + "{}_{}_{}_result".format(epoch,iteration,opt.model)):
            #     os.makedirs(opt.result_dir + "{}_{}_{}_result".format(epoch,iteration,opt.model))

            sr = sr.permute(0,2,3,1).detach().cpu().numpy()
            sr_0 = sr[0,:,:,:]

            scipy.misc.toimage(sr_0 * 255, high=255, low=0, cmin=0, cmax=255).save(
                    opt.result_dir + "/{}_{}_{}_result_{}.jpg".format(epoch,iteration,opt.model,iteration))
