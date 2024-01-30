from torch.nn import MSELoss, SmoothL1Loss, L1Loss
from dataset import Mayo_Dataset
from torch.utils.data import Dataset, DataLoader
from util import transforms
from util.util import create_optimizer,CharbonnierLoss
import torch
from trainer import train
from options.train_options import TrainOptions
import numpy as np
import os
from warmup_scheduler import GradualWarmupScheduler
from litformer import LITFormer




if __name__ == '__main__':

    GLOBAL_RANDOM_STATE = np.random.RandomState(47)
    seed = GLOBAL_RANDOM_STATE.randint(10000000)
    RandomState1=np.random.RandomState(seed)
    RandomState2=np.random.RandomState(seed)

    min_value=-1000
    max_value=2000

    train_raw_transformer=transforms.Compose([
    transforms.RandomFlip(RandomState1),
    transforms.RandomRotate90(RandomState1),
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    train_label_transformer=transforms.Compose([
    transforms.RandomFlip(RandomState2),
    transforms.RandomRotate90(RandomState2),
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    val_raw_transformer=transforms.Compose([
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    val_label_transformer=transforms.Compose([
    transforms.Normalize(min_value=min_value, max_value=max_value),
    transforms.ToTensor(expand_dims=False)
    ])

    train_transforms=[train_raw_transformer,train_label_transformer]
    val_transforms=[val_raw_transformer,val_label_transformer]

 
    opt = TrainOptions().parse()
    device=torch.device('cuda:{}'.format(opt.gpu_ids[0]) if torch.cuda.is_available() else "cpu")

    train_dataset=Mayo_Dataset(opt,transforms=train_transforms)
    train_dataloader=DataLoader(train_dataset,batch_size=opt.train_batch_size,shuffle=True,num_workers=8)
    if opt.is_val:
        opt.phase='test512'
        val_dataset=Mayo_Dataset(opt,transforms=val_transforms)
        val_dataloader=DataLoader(val_dataset,batch_size=opt.test_batch_size,shuffle=False,num_workers=4)


    model=LITFormer(in_channels=1,out_channels=1,n_channels=64,num_heads_s=[1,2,4,8],num_heads_t=[1,2,4,8],res=True,attention_s=True,attention_t=True).to(device)

    
    if len(opt.gpu_ids)>1:
        model=torch.nn.DataParallel(model,device_ids=opt.gpu_ids)


    loss_fn=CharbonnierLoss()

    optimizer=create_optimizer(opt,model)
    warmup_epochs=0
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs-warmup_epochs, eta_min=1e-6)
    #lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    lr_scheduler=scheduler_cosine


    train(
        opt, 
        model,
        optimizer,
        lr_scheduler,
        loss_fn,
        train_dataloader,
        val_dataloader,
        device=device,
        )


