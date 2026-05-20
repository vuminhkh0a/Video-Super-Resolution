'''
SOURCE CODE ARE COPIED FROM: https://github.com/junpan19/RSDN/blob/master/RSDN/arch.py
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class cha_loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(cha_loss, self).__init__()
        self.eps=eps
        return

    def forward(self, inp, target):
        diff = torch.abs(inp - target) ** 2 + self.eps ** 2
        out = torch.sqrt(diff)
        loss = torch.mean(out)

        return loss

def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, :, index]

    return it

class UPSCALE(nn.Module): 
    def __init__(self, basic_feature=64, scale=4, act=nn.LeakyReLU(0.2,True)):
        super(UPSCALE, self).__init__()
        body = []
        body.append(nn.Conv2d(basic_feature, 48, 3, 1, 3//2))
        body.append(act)
        body.append(nn.PixelShuffle(2))
        body.append(nn.Conv2d(12, 12, 3, 1, 3//2))
        body.append(nn.PixelShuffle(2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


class PFRB(nn.Module):
    '''
    Progressive Fusion Residual Block
    Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations, ICCV 2019
    '''
    def __init__(self, basic_feature=64, num_channel=3, act=torch.nn.LeakyReLU(0.2,True)):
        super(PFRB, self).__init__()
        self.bf = basic_feature
        self.nc = num_channel
        self.act = act
        self.conv0 = nn.Sequential(*[nn.Conv2d(self.bf, self.bf, 3, 1, 3//2) for _ in range(num_channel)])
        self.conv1 = nn.Conv2d(self.bf * num_channel, self.bf, 1, 1, 1//2)
        self.conv2 = nn.Sequential(*[nn.Conv2d(self.bf * 2, self.bf, 3, 1, 3//2) for _ in range(num_channel)])
    
    def forward(self, x):
        x1 = [self.act(self.conv0[i](x[i])) for i in range(self.nc)]
        merge = torch.cat(x1,1)
        base = self.act(self.conv1(merge))
        x2 = [torch.cat([base, i],1) for i in x1]
        x2 = [self.act(self.conv2[i](x2[i])) for i in range(self.nc)]

        return [torch.add(x[i], x2[i]) for i in range(self.nc)]

class UNIT(nn.Module): 
    def __init__(self, kind='successor', basic_feature=64, num_frame=3, num_b=5, scale=4, act=nn.LeakyReLU(0.2,True)):
        super(UNIT, self).__init__()
        self.bf = basic_feature
        self.nf = num_frame
        self.num_b = num_b
        self.scale = scale
        self.act = act
        self.kind = kind
        if kind == 'precursor':
            self.conv_c = nn.Conv2d(3, self.bf, 3, 1, 3//2)
            self.conv_sup = nn.Conv2d(3 * (num_frame-1), self.bf, 3, 1, 3//2)
        else:
            self.conv_c = nn.Sequential(*[nn.Conv2d((3 + self.bf), self.bf, 3, 1, 3//2) for i in range(num_frame)])
        self.blocks = nn.Sequential(*[PFRB(self.bf, 3, act) for i in range(num_b)])
        self.merge = nn.Conv2d(3 * self.bf, self.bf, 3, 1, 3//2)
        self.upscale = UPSCALE(self.bf, scale, act)
    
    def forward(self, it, ht_past, ht_now=None, ht_future=None):
        B, C, T, H, W = it.shape

        if self.kind == 'precursor':
            it_c = it[:, :, T // 2]
            index_sup = list(range(T))
            index_sup.pop(T // 2)
            it_sup = it[:, :, index_sup]
            it_sup = it_sup.view(B, C * (T - 1), H, W)
            hsup = self.act(self.conv_sup(it_sup))
            hc = self.act(self.conv_c(it_c))
            inp = [hc, hsup, ht_past]
        else:
            ht= [ht_past, ht_now, ht_future]
            it_c = [torch.cat([it[:, :, i, :, :], ht[i]], 1) for i in range(3)]
            inp = [self.act(self.conv_c[i](it_c[i])) for i in range(3)]

        inp = self.blocks(inp)

        ht = self.merge(torch.cat(inp, 1))
        it_sr = self.upscale(ht)

        return it_sr, ht

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.bf = config.model.basic_filter
        self.num_pb = config.model.num_pb
        self.num_sb = config.model.num_sb
        self.scale = config.model.scale
        self.nf = config.model.num_frame
        self.kind = config.model.kind     #local or global
        self.act = nn.LeakyReLU(0.2,True)
        self.precursor = UNIT('precursor', self.bf, self.nf, self.num_pb, self.scale, self.act)
        self.successor = UNIT('successor', self.bf, self.nf, self.num_sb, self.scale, self.act)

    def forward(self, x, start=0):
        B, C, T, H, W = x.shape
        start = max(0, start)
        end = T - start

        sr_all =  []
        pre_sr_all =  []
        pre_ht_all = []
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device = x.device)

        # precursor
        for idx in range(T):
            t = idx if self.kind == 'local' else T - idx - 1
            insert_idx = T + 1 if self.kind == 'local' else 0
            
            it = generate_it(x, t, self.nf, T)
            it_sr_pre, ht_past= self.precursor(it, ht_past, None, None)
            pre_ht_all.insert(insert_idx, ht_past)
            pre_sr_all.insert(insert_idx, it_sr_pre)

        #successor
        ht_past = torch.zeros((B, self.bf, H, W), dtype=torch.float, device = x.device)
        for t in range(end):
            it = generate_it(x, t, self.nf, T)
            ht_future = pre_ht_all[t] if t == T - 1 else pre_ht_all[t + 1]
            it_sr, ht_past= self.successor(it, ht_past, pre_ht_all[t], ht_future)
            sr_all.append(it_sr + pre_sr_all[t])

        sr_all = torch.stack(sr_all, 2)[:, :, start:]
        pre_sr_all = torch.stack(pre_sr_all, 2)[:, :, start:end]
        
        return sr_all, pre_sr_all