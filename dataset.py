#dataset
import os, glob, shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from util import transforms
import ipdb
import random
random.seed(42)

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

def random_sample(input_list, sample_size):
    if sample_size > len(input_list):
        sample_size = len(input_list)
    return random.sample(input_list, sample_size)


class Mayo_Dataset(Dataset):
    def __init__(self, opt,transforms=None):
        #ipdb.set_trace()
        self.transforms = transforms
        #hu_min, hu_max = hu_range
        self.phase=opt.phase
        self.mirror_padding=opt.mirror_padding

        self.q_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/quarter_lr/*')
        self.f_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/full_hr/*')


    def __getitem__(self, index):
        f_data=np.load(self.f_path_list[index]).astype(np.float32)
        q_data = np.load(self.q_path_list[index]).astype(np.float32)

        if self.transforms is not None:
            f_data = self.transforms[1](f_data)
            q_data = self.transforms[0](q_data)
        return q_data, f_data

    def __len__(self):
        return len(self.q_path_list)
    
    