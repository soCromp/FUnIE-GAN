"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan_up import GeneratorFunieGANUP, DiscriminatorFunieGANUP, DiscriminatorFunieGANP
from utils.data_utils import GetTrainingData, GetValImage

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="configs/train_euvp.yaml")
#parser.add_argument("--cfg_file", type=str, default="configs/train_ufo.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=201, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2 
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
# dataset_name = cfg["dataset_name"] 
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]


## create dir for model and validation data
samples_dir = os.path.join("samples/FunieGAN/up/euvp")
checkpoint_dir = os.path.join("checkpoints/FunieGAN/up/euvp")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


""" FunieGAN specifics: loss functions and patch-size
-----------------------------------------------------"""
mse = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
L_vgg = VGG19_PercepLoss() # content loss (vgg)
lambda_1, lambda_con = 7, 3 # 7:3 (as in paper)
patch = (1, img_height//16, img_width//16) # 16x16 for 256x256

# Initialize generator and discriminator
Gc = GeneratorFunieGANUP() # clear generator
Dc = DiscriminatorFunieGANUP() # clear discriminator for unpaired data
DPc = DiscriminatorFunieGANP() # clear discriminator for paired data
Gu = GeneratorFunieGANUP()  # unclear generator
Du = DiscriminatorFunieGANUP() # unclear discriminator
DPu = DiscriminatorFunieGANP() # unclear discriminator for paired data

# see if cuda is available
if torch.cuda.is_available():
    Gc = Gc.cuda()
    Dc = Dc.cuda()
    DPc=DPc.cuda()
    Gu = Gu.cuda()
    Du = Du.cuda()
    DPu=DPu.cuda()
    mse.cuda()
    mae.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    Gc.apply(Weights_Normal)
    Dc.apply(Weights_Normal)
    DPc.apply(Weights_Normal)
    Gu.apply(Weights_Normal)
    Du.apply(Weights_Normal)
    DPu.apply(Weights_Normal)
else:
    print('train_funiegan.py line 96')
    exit
#     generator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
#     discriminator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
#     print ("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_Gc = torch.optim.Adam(Gc.parameters(),  lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_Dc = torch.optim.Adam(Dc.parameters(),  lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_DPc= torch.optim.Adam(DPc.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_Gu = torch.optim.Adam(Gu.parameters(),  lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_Du = torch.optim.Adam(Du.parameters(),  lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_DPu= torch.optim.Adam(DPu.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader_up = DataLoader( # unpaired data
    GetTrainingData(dataset_path, 'unpaired', transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2,
)

dataloader_p = DataLoader( # paired data
    GetTrainingData(dataset_path, 'paired', transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)


def train_unpaired():
    for i, batch in enumerate(dataloader_up):
        # Model inputs
        imgA = Variable(batch["A"].type(Tensor)) #unclear
        imgB = Variable(batch["B"].type(Tensor)) #clear
        # Adversarial ground truths
        ones = Variable(Tensor(np.ones((imgA.size(0), *patch))), requires_grad=False)
        zeroes = Variable(Tensor(np.zeros((imgA.size(0), *patch))), requires_grad=False)

        optimizer_Du.zero_grad()
        optimizer_Dc.zero_grad()
        fakeB = Gc(imgA)
        fakeA = Gu(imgB)

        #discriminator loss

        pred_realDu = Du(imgA)
        loss_realDu = mse(pred_realDu, ones)
        pred_fakeDu = Du(fakeA)
        loss_fakeDu = mse(pred_fakeDu, zeroes)
        # Total loss: real + fake (standard PatchGAN)
        loss_Du = 0.5 * (loss_realDu + loss_fakeDu) * 10.0 # 10x scaled for stability
        
        pred_realDc = Dc(imgB)
        loss_realDc = mse(pred_realDc, ones)
        pred_fakeDc = Dc(fakeB)
        loss_fakeDc = mse(pred_fakeDc, zeroes)
        loss_Dc = 0.5 * (loss_realDc + loss_fakeDc) * 10.0

        loss_Du.backward()
        loss_Dc.backward()
        optimizer_Du.step()
        optimizer_Dc.step()

        optimizer_Gc.zero_grad()
        optimizer_Gu.zero_grad()

        fakeB = Gc(imgA)
        fakeA = Gu(imgB)
        pred_fakeDu = DPu.forward(imgB, fakeA)
        pred_fakeDu = Du(fakeA)
        pred_fakeDc = Dc(fakeB)
        loss_foolGu = mse(pred_fakeDu, ones) #since generator wants to fool discriminator
        loss_foolGc = mse(pred_fakeDc, ones)

        reconstrA = Gu(fakeB)
        reconstrB = Gc(fakeA)
        loss_reconA = mae(reconstrA, imgA)
        loss_reconB = mae(reconstrB, imgB)

        idA = Gu(imgA) #generate clear from real clear - should be identical
        idB = Gc(imgB)
        loss_idA = mae(idA, imgA)
        loss_idB = mae(idB, imgB)

        # loss
        lossG = loss_foolGu + loss_foolGc + 10*loss_reconA + 10*loss_reconB + loss_idA + loss_idB
        lossG.backward()
        optimizer_Gc.step()
        optimizer_Gu.step()

    return loss_Du.detach(), loss_Dc.detach(), lossG.detach() #stats for last batch


def train_paired():
    for i, batch in enumerate(dataloader_p):
        # Model inputs
        imgA = Variable(batch["A"].type(Tensor)) #unclear
        imgB = Variable(batch["B"].type(Tensor)) #clear
        # Adversarial ground truths
        ones = Variable(Tensor(np.ones((imgA.size(0), *patch))), requires_grad=False)
        zeroes = Variable(Tensor(np.zeros((imgA.size(0), *patch))), requires_grad=False)

        optimizer_DPu.zero_grad()
        optimizer_DPc.zero_grad()
        fakeB = Gc(imgA)
        fakeA = Gu(imgB)

        pred_realDPu = DPu(imgB, imgA)
        pred_realDPc = DPc(imgA, imgB)
        loss_realDPu = mse(pred_realDPu, ones)
        loss_realDPc = mse(pred_realDPc, ones)

        pred_fakeDPu = DPu(imgB, fakeA)
        pred_fakeDPc = DPc(imgA, fakeB)
        loss_fakeDPu = mse(pred_fakeDPu, zeroes)
        loss_fakeDPc = mse(pred_fakeDPc, zeroes)

        loss_DPu = 0.5 * (loss_realDPu + loss_fakeDPu) * 10
        loss_DPc = 0.5 * (loss_fakeDPc + loss_fakeDPc) * 10
        
        loss_DPu.backward()
        loss_DPc.backward()
        optimizer_DPu.step()
        optimizer_DPc.step()

        optimizer_Gc.zero_grad()
        optimizer_Gu.zero_grad()

        fakeB = Gc(imgA)
        fakeA = Gu(imgB)
        pred_fakeDPu = DPu(imgB, fakeA)
        pred_fakeDPc = DPc(imgA, fakeB)
        loss_foolGu = mse(pred_fakeDPu, ones) # since generator wants to fool discriminator
        loss_foolGc = mse(pred_fakeDPc, ones)

        loss_1u = mae(fakeB, imgB)
        loss_conu = L_vgg(fakeB, imgB)
        loss_1c = mae(fakeA, imgA)
        loss_conc = L_vgg(fakeA, imgA)
        
        loss_Gu = loss_foolGu + lambda_1*loss_1u + lambda_con*loss_conu
        loss_Gc = loss_foolGc + lambda_1*loss_1c + lambda_con*loss_conc

        loss_Gu.backward()
        loss_Gc.backward()
        optimizer_Gu.step()
        optimizer_Gc.step()

    return loss_DPu.detach(), loss_DPc.detach(), 0.5 * (loss_Gu.detach() + loss_Gc.detach()) #stats for last batch


## Training pipeline
for epoch in range(epoch, num_epochs):
    #call train_paired or train_unpaired
    loss_Du, loss_Dc, lossG = train_paired()

    ## Print log
    sys.stdout.write("\r[Epoch %d/%d] [DuLoss: %.3f, DcLoss: %.3f, GLoss: %.3f]"
                        %(
                        epoch, num_epochs,
                        loss_Du.item(), loss_Dc.item(), lossG.item(),
                        )
    )
    ## If at sample interval save image
    if epoch % val_interval == 0:
        imgs = next(iter(val_dataloader))
        imgs_val = Variable(imgs["val"].type(Tensor))
        imgs_gen = Gc(imgs_val)
        img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
        save_image(img_sample, "samples/FunieGAN/up/%s.png" % (epoch), nrow=5, normalize=True)

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        torch.save(Gu.state_dict(), "checkpoints/FunieGAN/up/generatorU_%d.pth" % (epoch))
        torch.save(Du.state_dict(), "checkpoints/FunieGAN/up/discriminatorU_%d.pth" % (epoch))
        torch.save(DPu.state_dict(), "checkpoints/FunieGAN/up/discriminatorUpaired_%d.pth" % (epoch))
        torch.save(Gc.state_dict(), "checkpoints/FunieGAN/up/generatorC_%d.pth" % (epoch))
        torch.save(Dc.state_dict(), "checkpoints/FunieGAN/up/discriminatorC_%d.pth" % (epoch))
        torch.save(DPc.state_dict(), "checkpoints/FunieGAN/up/discriminatorCpaired_%d.pth" % (epoch))
