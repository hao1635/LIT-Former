'''
LIT-Former
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import ipdb
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
from torch.nn import init
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): 
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        

class eMSM_T(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_T, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.position_embedding=PositionalEncoding(d_model=dim)
        
        self.project_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(0.)
        )

    def forward(self, x):
        b,c,t,h,w=x.shape

        x=F.adaptive_avg_pool3d(x,(t,1,1))

        x=x.squeeze(-1).squeeze(-1).permute(2,0,1) #t,b,c

        x= self.position_embedding(x).permute(1,0,2) #b,t,c

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        #ipdb.set_trace()

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads), (q, k, v))

        scale = (c//self.num_heads) ** -0.5
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.num_heads)

        out = self.project_out(out).permute(0,2,1)

        return out


class eMSM_I(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(eMSM_I, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,t,h,w = x.shape
        x=F.adaptive_avg_pool3d(x,(1,h,w))
        x=x.permute(0,1,3,4,2).squeeze(-1)

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

         
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

class LITFormerBlock(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads_s=8,num_heads_t=2,kernel_size=1,stride=1,padding=0,
                groups=1,bias=False,res=True,attention_s=False,attention_t=False):
        super().__init__()
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        assert len(kernel_size) == len(stride) == len(padding) == 3
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        self.res=res
        self.attn_s=attention_s
        self.attn_t=attention_t
        self.num_heads_s=num_heads_s
        self.num_heads_t=num_heads_t
        self.activation=nn.LeakyReLU(inplace=True)

        if attention_s==True:
            self.attention_s=eMSM_I(dim=input_channel, num_heads=num_heads_s, bias=False)
        self.conv_1x3x3=nn.Conv3d(input_channel,output_channel,kernel_size=(1, kernel_size[1], kernel_size[2]),
                            stride=(1, stride[1], stride[2]),padding=(0, padding[1], padding[2]),groups=groups)
        if attention_t==True:
            self.attention_t=eMSM_T(dim=input_channel, num_heads=num_heads_t, bias=False)
        self.conv_3x1x1=nn.Conv3d(input_channel,output_channel,kernel_size=(kernel_size[0], 1, 1),
                            stride=(stride[0], 1, 1),padding=(padding[0], 0, 0),groups=groups)

        if self.input_channel != self.output_channel:
            self.shortcut=nn.Conv3d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)


    def forward(self, inputs):

        if self.attn_s==True or self.attn_t==True:

            attn_s=self.attention_s(inputs).unsqueeze(2)  if self.attn_s==True else 0 
            attn_t=self.attention_t(inputs).unsqueeze(-1).unsqueeze(-1) if self.attn_t==True else 0

            inputs_attn=inputs+attn_t+attn_s

            conv_S=self.conv_1x3x3(inputs_attn)
            conv_T=self.conv_3x1x1(inputs_attn)

            if self.input_channel == self.output_channel: 
                identity_out=inputs_attn 
            else: 
                identity_out=self.shortcut(inputs_attn)

        else:
            if self.input_channel == self.output_channel: 
                identity_out=inputs 
            else: 
                identity_out=self.shortcut(inputs)
                
            conv_S=self.conv_1x3x3(inputs)
            conv_T=self.conv_3x1x1(inputs)

        if self.res:
            output=conv_S+conv_T+identity_out
        elif not self.res:
            output=conv_S+conv_T  

        return output


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super(DoubleConv,self).__init__()
        self.double_conv=nn.Sequential(
            LITFormerBlock(in_channels,in_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,res=res,
                            attention_s=attention_s,attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            LITFormerBlock(in_channels,out_channels,res=res),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads_s=8,num_heads_t=2,
                 res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d((1,2,2), (1,2,2)),
            DoubleConv(in_channels, out_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,
                       res=res,attention_s=attention_s,attention_t=attention_t)
        )
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool3d((1,2,2), (1,2,2)),
            LITFormerBlock(in_channels,2*in_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,res=res,
                      attention_s=attention_s,attention_t=attention_t),
            nn.LeakyReLU(inplace=True),
            LITFormerBlock(2*in_channels,out_channels),
            nn.LeakyReLU(inplace=True),
            )
    def forward(self, x):
        return self.encoder(x)


    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True,trilinear=True, num_heads_s=8,num_heads_t=2,
                res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.res_unet=res_unet
        if trilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels , kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels,num_heads_s=num_heads_s,num_heads_t=num_heads_t,
                               res=res,attention_s=attention_s,attention_t=attention_t)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,res=True,activation=False):
        super().__init__()
        self.act=activation
        self.conv =LITFormerBlock(in_channels, out_channels,res=res)
        self.activation = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        x=self.conv(x)
        if self.act==True:
            x=self.activation(x)
        return x
        
        
class LITFormer(nn.Module):
    def __init__(self, in_channels,out_channels,n_channels,num_heads_s=[1,2,4,8],num_heads_t=[1,2,4,8],
                res=True,attention_s=False,attention_t=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation=True)
        self.enc1 = DoubleConv(n_channels//2, n_channels,num_heads_s=num_heads_s[0],num_heads_t=num_heads_t[0],
                               res=res,attention_s=attention_s,attention_t=attention_t) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads_s=num_heads_s[1],num_heads_t=num_heads_t[1],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads_s=num_heads_s[2],num_heads_t=num_heads_t[2],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads_s=num_heads_s[3],num_heads_t=num_heads_t[3],
                             res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads_s=num_heads_s[2],num_heads_t=num_heads_t[2],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads_s=num_heads_s[1],num_heads_t=num_heads_t[1],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads_s=num_heads_s[0],num_heads_t=num_heads_t[0],
                               res=res,attention_s=attention_s,attention_t=attention_t)
        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation=True)
        self.depth_up = nn.Upsample(scale_factor=tuple([2.5,1,1]),mode='trilinear')
        self.out2 = SingleConv(n_channels//2,out_channels,res=res,activation=False)

    def forward(self, x):
        b,c,d,h,w=x.shape
        x =self.firstconv(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output)+x
        output = self.depth_up(output)
        output = self.out2(output)
        return output

