import torch
import torch.nn as nn
import torch.nn.functional as F

from mmagic.models.editors.basicvsr import BasicVSRNet
from mmagic.models.editors.edvr import EDVRNet
from rsdn import RSDN9_128
from ovsr import Net

NUM_FRAMES = 41
CENTER_FRAME_IDX = NUM_FRAMES // 2   # 20


# BasicVSR

class BasicVSRWrapper(nn.Module):

    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()

        self.model = BasicVSRNet(mid_channels=64, num_blocks=30, spynet_pretrained=None)

        # Fix loading weight error from mmagic (change prefix generator.spynet to spynet)

        if checkpoint_path is not None:

            ckpt = torch.load(checkpoint_path, map_location='cpu')

            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            new_state_dict = {}

            for k, v in state_dict.items():

                prefixes = ['generator.', 'module.', 'net_g.', 'model.',]

                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]

                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict,strict=False)

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


#EDVR

class EDVRWrapper(nn.Module):

    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()

        self.num_frames = 5
        self.radius = self.num_frames // 2

        self.model = EDVRNet(in_channels=3, out_channels=3, mid_channels=128, num_frames=self.num_frames, \
            center_frame_idx=self.radius, deform_groups=8, num_blocks_extraction=5, num_blocks_reconstruction=20, with_tsa=True)

        # Fix loading weight error from mmagic (change prefix generator.spynet to spynet)

        if checkpoint_path is not None:

            ckpt = torch.load(checkpoint_path, map_location='cpu')

            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            new_state_dict = {}

            for k, v in state_dict.items():

                prefixes = ['generator.', 'module.', 'net_g.', 'model.',]

                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]

                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False)

        self.model.eval()
        self.model.to(device)

    def get_window(self, x, idx):
        """
        x:
            (B, T, C, H, W)

        return:
            (B, num_frames, C, H, W)
        """

        B, T, C, H, W = x.shape
        start = max(0, idx - self.radius)
        end = min(T, idx + self.radius + 1)

        clip = x[:, start:end]

        if clip.shape[1] < self.num_frames:
            pad_left = max(0, self.radius - idx)
            pad_right = self.num_frames - clip.shape[1] - pad_left
            
            if pad_left > 0:
                left = clip[:, 0:1].repeat(1, pad_left, 1, 1, 1)
                clip = torch.cat([left, clip], dim=1)

            if pad_right > 0:
                right = clip[:, -1:].repeat(1, pad_right, 1, 1, 1)
                clip = torch.cat([clip, right], dim=1)

        return clip

    @torch.no_grad()
    def forward(self, x):

        B, T, C, H, W = x.shape
        outputs = []
        for i in range(T):
            clip = self.get_window(x, i)
            out = self.model(clip)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        return outputs
    

#RSDN
class RSDNWrapper(nn.Module):

    def __init__(self, checkpoint_path=None, scale=4, device='cuda'):
        super().__init__()

        self.scale = scale
        self.model = RSDN9_128(scale)

        if checkpoint_path is not None:

            ckpt = torch.load(
                checkpoint_path,
                map_location='cpu'
            )
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            # Fix loading weight error by removing module prefix

            new_state_dict = {}
            for k, v in state_dict.items():
                prefixes = ['module.',]
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                new_state_dict[k] = v

            self.model.load_state_dict(
                new_state_dict,
                strict=False
            )

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def forward(self, x):

        B, T, C, H, W = x.shape

        S = F.interpolate(
            x.view(B*T, C, H, W),
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        )
        S = F.interpolate(
            S,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        S = S.view(B, T, C, H, W)
        D = x - S

        out, _, _ = self.model(x, D, S)

        return out
    

# OVSR
class OVSRWrapper(nn.Module):

    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()
        class Config:
            pass
        class ModelConfig:
            pass
        config = Config()
        config.model = ModelConfig()

        config.model.basic_filter = 56
        config.model.num_pb = 4
        config.model.num_sb = 2
        config.model.scale = 4
        config.model.num_frame = 3
        config.model.kind = 'global'

        self.model = Net(config)

        if checkpoint_path is not None:

            ckpt = torch.load(checkpoint_path, map_location='cpu')

            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            new_state_dict = {}
            for k, v in state_dict.items():
                prefixes = ['module.',]
                for p in prefixes:
                    if k.startswith(p):
                        k = k[len(p):]
                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=True)

        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        sr, _ = self.model(x)
        sr = sr.permute(0, 2, 1, 3, 4).contiguous()
        return sr