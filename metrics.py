import torch
import torch.nn.functional as F


def metric_psnr(pred, gt, max_val=1.0):
    """
    pred, gt: [B, T, C, H, W], range [0,1]
    return: average PSNR over all frames
    """
    B, T, C, H, W = pred.shape
    
    mse = F.mse_loss(pred, gt, reduction='none')
    mse = mse.view(B, T, -1).mean(dim=2)

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr.mean().item()

def gaussian_window(window_size=11, sigma=1.5, channel=3):
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    window = window.expand(channel, 1, window_size, window_size)
    return window

def metric_ssim(pred, gt, window_size=11):
    """
    pred, gt: [B, T, C, H, W]
    """
    B, T, C, H, W = pred.shape
    pred = pred.view(B*T, C, H, W)
    gt = gt.view(B*T, C, H, W)

    window = gaussian_window(window_size, channel=C).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(gt, window, padding=window_size//2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size//2, groups=C) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()