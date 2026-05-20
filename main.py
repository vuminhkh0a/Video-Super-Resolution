import time
import torch
from tqdm import tqdm

from data import get_loader
from metrics import metric_psnr, metric_ssim

from model import *


def evaluate(model, loader, device):
    '''
    Evaluate a video super-resolution model on a dataset.

    The evaluation computes:
        - PSNR (Peak Signal-to-Noise Ratio)
        - SSIM (Structural Similarity Index)
        - FPS (Frames Per Second)
        - Latency per frame (ms)

    Args:
        model (nn.Module):
            Video super-resolution model.

        loader (DataLoader):
            PyTorch DataLoader that returns:
                bd : degraded blurry sequence
                bi : degraded bicubic sequence
                gt : ground-truth high-resolution sequence

        device (str or torch.device):
            Device used for inference (e.g., 'cuda:0' or 'cpu').

    Returns:
        dict:
            {
                'PSNR'   : average PSNR over all sequences,
                'SSIM'   : average SSIM over all sequences,
                'FPS'    : inference throughput (frames per second),
                'Latency': average inference latency per frame in milliseconds
            }
    '''
    total_psnr = 0
    total_ssim = 0
    total_time = 0
    total_frames = 0
    count = 0

    model.eval()

    for bd, bi, gt in tqdm(loader):

        bi = bi.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(bi)
            end_time = time.time()

        inference_time = end_time - start_time
        total_time += inference_time
        n_frames = bi.shape[1]
        total_frames += n_frames

        psnr = metric_psnr(pred, gt)
        ssim = metric_ssim(pred, gt)
        total_psnr += psnr
        total_ssim += ssim
        count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    fps = total_frames / total_time
    latency_per_frame = (total_time / total_frames) * 1000  

    return {'PSNR': avg_psnr, 'SSIM': avg_ssim, 'FPS': fps, 'Latency': latency_per_frame,}


def main():

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    loader = get_loader(batch_size=1, num_workers=4, pin_memory=True)
    models = {
        'BasicVSR': BasicVSRWrapper(checkpoint_path='./checkpoints/vsrnet_checkpoints/basicvsr_vimeo90k_bi_20210409-d2d8f760.pth', device=device),
        'EDVR': EDVRWrapper(checkpoint_path='./checkpoints/edvr_checkpoints/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104-4509865f.pth', device=device),
        'RSDN': RSDNWrapper(checkpoint_path='./checkpoints/rsdn_checkpoints/RSDN.pth', device=device),
        'OVSR': OVSRWrapper(checkpoint_path='./checkpoints/ovsr_checkpoints/0721.pth', device=device)
    }

    for name, model in models.items():
        results = evaluate(model=model, loader=loader, device=device)
        print(f"\nModel: {name}")
        print(f"PSNR               : {results['PSNR']:.4f}")
        print(f"SSIM               : {results['SSIM']:.4f}")
        print(f"FPS                : {results['FPS']:.2f}")
        print(f"Latency per frame  : {results['Latency']:.2f} ms")

if __name__ == "__main__":
    main()