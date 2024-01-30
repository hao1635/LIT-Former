import os
import matplotlib.pyplot as plt
import torchvision
import torch 
import torch.nn as nn
from torch import optim
import logging
import shutil
from torchmetrics.functional import structural_similarity_index_measure
from kornia.filters import get_gaussian_kernel2d, filter2d
import torch.nn.functional as F

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(path, refresh=False):
    
    """ function for making directory (to save results). """
    
    try: os.mkdir(path)
    except: 
        if(refresh): 
            shutil.rmtree(path)
            os.mkdir(path)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def save_images(images,root,phase,index,normalize=False):
    numbers=images.shape[2]
    images=images[0].permute(1,0,2,3)
    saveroot=root+'/'+str('%02d' % index)+'-'+phase+'.png'
    torchvision.utils.save_image(images,saveroot,padding = 0,normalize=normalize)


def create_optimizer(opt,model):
    learning_rate = opt.learning_rate
    weight_decay =opt.weight_decay
    betas = opt.betas
    # weight_decay = optimizer_config.get('weight_decay', 0)
    # betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay,amsgrad=False)
    return optimizer
'''
torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, 
amsgrad=False, *, maximize=False, foreach=None, capturable=False)
'''

def crop_center(img,cropx,cropy,cropz):
    z,y,x = img.shape[2],img.shape[3],img.shape[4]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)  
    startz = z//2-(cropz//2)
    return img[:,:,startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]



def ssim_xy(input,target):
    assert input.size() == target.size()
    b,c,z,x,y=input.shape
    input=input.reshape(-1,1,x,y)
    target=target.reshape(-1,1,x,y)
    return structural_similarity_index_measure(input,target)

def compute_psnr2D(input: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    b,c,z,x,y=input.shape
    input=input.reshape(-1,1,x,y)
    target=target.reshape(-1,1,x,y)

    mse_val = F.mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


def compute_ssim(img1, img2, window_size=11, reduction: str = "mean", max_val: float = 1.0, full: bool = False):
    window: torch.Tensor = get_gaussian_kernel2d(
        (window_size, window_size), (1.5, 1.5))
    window = window.requires_grad_(False)
    assert img1.size() == img2.size()
    b,c,z,x,y=img1.shape
    img1=img1.reshape(-1,1,x,y)
    img2=img2.reshape(-1,1,x,y)
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2
    tmp_kernel: torch.Tensor = window.to(img1)
    tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)
    # compute local mean per channel
    mu1: torch.Tensor = filter2d(img1, tmp_kernel)
    mu2: torch.Tensor = filter2d(img2, tmp_kernel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # compute local sigma per channel
    sigma1_sq = filter2d(img1 * img1, tmp_kernel) - mu1_sq
    sigma2_sq = filter2d(img2 * img2, tmp_kernel) - mu2_sq
    sigma12 = filter2d(img1 * img2, tmp_kernel) - mu1_mu2

    ssim_map = ((2. * mu1_mu2 + C1) * (2. * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = ssim_map
    if reduction != 'none':
        ssim_score = torch.clamp(ssim_score, min=0, max=1)
        if reduction == "mean":
            ssim_score = torch.mean(ssim_score)
        elif reduction == "sum":
            ssim_score = torch.sum(ssim_score)
    if full:
        cs = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        return ssim_score, cs
    return ssim_score


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, t,h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w*t)

def compute_rmse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(input, target))