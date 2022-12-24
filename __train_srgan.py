import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

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
from losses import CharbonnierLoss, GeneratorLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler

# thuytt
from networks.model_srgan import Discriminator
from torch.autograd import Variable


######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES


# thuytt test
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler('myLog.log')
logger.addHandler(fh)

import neptune.new as neptune
run = neptune.init(
    project="thuy4tbn99/MIRNet",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmMDZlN2UwYS1lODc1LTRlMTctYjQzNS00MmEwOTJiZWU5YzIifQ==",
)


if __name__=='__main__':
    NUM_EPOCHS = opt.OPTIM.NUM_EPOCHS
    ######### Model ###########
    netG = MIRNet()
    netD = Discriminator()
    netG.cuda()
    netD.cuda()    

    device_ids = [i for i in range(torch.cuda.device_count())]
    Tensor = torch.cuda.FloatTensor if torch.cuda.device_count() else torch.FloatTensor
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


    new_lr = opt.OPTIM.LR_INITIAL

    optimizerG = optim.Adam(netG.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8, weight_decay=1e-8)
    optimizerD = optim.Adam(netD.parameters())

    ######### Scheduler ###########
    warmup =False
    if warmup:
        warmup_epochs = 3
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizerG, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizerG, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()

    ######### Resume ###########
    """
    if opt.TRAINING.RESUME:
        path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')
    """

    if len(device_ids)>1:
        netG = nn.DataParallel(netG, device_ids = device_ids)
        netD = nn.DataParallel(netD, device_ids = device_ids)

    ######### Loss ###########
    # criterion = CharbonnierLoss().cuda()
    generatorLoss = GeneratorLoss().cuda()
    adversarialLoss = torch.nn.BCELoss().cuda()

    ######### DataLoaders ###########
    img_options_train = {'patch_size':opt.TRAINING.TRAIN_PS}

    train_dataset = get_training_data(train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)

    val_dataset = get_validation_data(val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    mixup = utils.MixUp_AUG()
    best_psnr = 0
    best_epoch = 0
    best_iter = 0

    eval_now = len(train_loader)//10 - 1
    # eval_now=10
    print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
            
        # for i, data in enumerate(tqdm(train_loader), 0):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for i, data in enumerate(train_bar, 0):
            batch_size = opt.OPTIM.BATCH_SIZE
            running_results['batch_sizes'] += batch_size
            # zero_grad
            for param in netG.parameters():
                param.grad = None
            
            target = data[0].cuda()
            input_ = data[1].cuda()

            # if epoch>5:
            #     target, input_ = mixup.aug(target, input_)

            # restored = model_restoration(input_)
            # restored = torch.clamp(restored,0,1)  
            real_img = target
            fake_img = netG(input_)
            # fake_img = torch.clamp(fake_img,0,1)  

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # print('real_out', real_out.shape, real_out)
            # print('fake_out', fake_out.shape, fake_out)
            # print('d_loss', d_loss)
            

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            fake_img = netG(input_)
            fake_out = netD(fake_img).mean()

            g_loss = generatorLoss(fake_out, fake_img, real_img)
            g_loss.backward()
            optimizerG.step()

            run['train_Gloss'].log(g_loss.item())
            run['train_Dloss'].log(d_loss.item())

            # epoch_loss +=g_loss.item()
            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
            
            
            #### Evaluation ####
            if i%eval_now==0 and i>0:
                netG.eval()
                with torch.no_grad():
                    psnr_val_rgb = []
                    val_bar = tqdm(val_loader)
                    for ii, data_val in enumerate(val_bar, 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        filenames = data_val[2]

                        restored = netG(input_)
                        restored = torch.clamp(restored,0,1) 
                        psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                        if ii%50==0:
                            x = utils.batch_PSNR(restored, target, 1.)
                            print(f'i: {ii} -- x: ', type(x), x)

                    psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
                    
                    if psnr_val_rgb > best_psnr:
                        print(f'Bingo save best\n best_psnr: {best_psnr} -- after: {psnr_val_rgb}')
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch
                        best_iter = i 
                        torch.save({'epoch': epoch, 
                                    'state_dict': netG.state_dict(),
                                    'optimizer' : optimizerG.state_dict()
                                    }, os.path.join(model_dir,"netG_best.pth"))
                        torch.save({'epoch': epoch, 
                            'state_dict': netD.state_dict(),
                            'optimizer' : optimizerD.state_dict()
                            }, os.path.join(model_dir,"netD_best.pth")) 
                        

                    print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
                
                netG.train()

   
        
        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time,epoch_loss))
        print("------------------------------------------------------------------")

        

        torch.save({'epoch': epoch, 
                    'state_dict': netG.state_dict(),
                    'optimizer' : optimizerG.state_dict()
                    }, os.path.join(model_dir,"netG_latest.pth"))   
        
        torch.save({'epoch': epoch, 
                    'state_dict': netD.state_dict(),
                    'optimizer' : optimizerD.state_dict()
                    }, os.path.join(model_dir,"netD_latest.pth"))   

