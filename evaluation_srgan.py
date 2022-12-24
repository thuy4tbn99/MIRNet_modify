import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True
import sys

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data
from pdb import set_trace as stx

from networks.MIRNet_model import MIRNet
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler


# thuytt
from networks.model_srgan import Discriminator

from PIL import Image
import torchvision.transforms as transforms
from transform_func import AddGaussianNoise, AddSnow, BrightnessTransform, MotionBlur
addGaussianNoise = AddGaussianNoise() 
addSnow = AddSnow()
brightness = BrightnessTransform(1.6)
motionBlur = MotionBlur()

# arguments
import argparse
parser = argparse.ArgumentParser(description='Evaluation on the validation set of SIDD')
parser.add_argument('--save_dir', default='test',
    type=str, help='Directory to save image')
parser.add_argument('--model_g', default='model_name.pth',
    type=str, help='Model Generator (.pth file)')
parser.add_argument('--model_d', default='model_name.pth',
    type=str, help='Model Discriminator (.pth file)')


args = parser.parse_args()
save_dir = args.save_dir
model_g = args.model_g
model_d = args.model_d
print('save_dir:', save_dir)
print('model_g:', model_g)
print('model_d:', model_d)


if __name__=='__main__':
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION
    val_dir   = opt.TRAINING.VAL_DIR
    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
    save_images = opt.TRAINING.SAVE_IMAGES

    ######### Model ###########
    netG = MIRNet()
    netD = Discriminator()
    netG.cuda()
    netD.cuda() 


    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    if len(device_ids)>1:
        netG = nn.DataParallel(netG, device_ids = device_ids)
        netD = nn.DataParallel(netD, device_ids = device_ids)

    new_lr = opt.OPTIM.LR_INITIAL
    optimizerG = optim.Adam(netG.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
    optimizerD = optim.Adam(netD.parameters())
    
    # load best model
    path_g = 'checkpoints/Denoising/models/MIRNet/' + model_g
    utils.load_checkpoint(netG, path_g)
    utils.load_optim(optimizerG, path_g)

    path_d = 'checkpoints/Denoising/models/MIRNet/' + model_d
    utils.load_checkpoint(netD, path_d)
    utils.load_optim(optimizerD, path_d)

    ######### DataLoaders ###########
    # transform
    transformList = []
    transformList.append(motionBlur)
    transformSequence=transforms.Compose(transformList)


    val_dataset = get_validation_data(val_dir, transformSequence)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)
    print('===> Loading datasets')

    netG.eval()
    netD.eval()
    with torch.no_grad():
        psnr_val_rgb = []
        val_bar = tqdm(val_loader)
        if save_images:
            save_dir = result_dir + '/' + save_dir
            utils.mkdir(save_dir)
        for ii, data_val in enumerate(val_bar, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            filenames = data_val[2]

            restored = netG(input_)
            restored = torch.clamp(restored,0,1) 
            psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

            psnr_value = utils.batch_PSNR(restored, target, 1.)
            print(f'i: {ii} -- x: ', type(psnr_value), psnr_value)

            if save_images:
                target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                
                for batch in range(input_.shape[0]):
                    temp = np.concatenate((input_[batch]*255, restored[batch]*255, target[batch]*255),axis=1) # restore image from normalize
                    if psnr_value.item()>=40:
                        utils.save_img(os.path.join(save_dir, 'high_' + filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))
                    else:
                        utils.save_img(os.path.join(save_dir, 'low_' + filenames[batch][:-4] +'.jpg'),temp.astype(np.uint8))
       

        psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
        print(f"PSNR SIDD: {psnr_val_rgb}")