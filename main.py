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



def evaluate(model, loader, device):
    '''
    Evaluate a video super-resolution model on both:
        - BI (bicubic degradation)
        - BD (blur-down degradation)

    Metrics:
        - PSNR
        - SSIM
        - FPS
        - Latency per frame

    Args:
        model (nn.Module):
            Video super-resolution model.

        loader (DataLoader):
            Returns:
                bd : blurry-downsampled sequence
                bi : bicubic-downsampled sequence
                gt : ground-truth HR sequence

        device:
            cuda or cpu

    Returns:
        {
            'BI': {...},
            'BD': {...}
        }
    '''

    model.eval()

    results = {
        'BI': {
            'PSNR': 0,
            'SSIM': 0,
            'TIME': 0,
            'FRAMES': 0,
            'COUNT': 0
        },
        'BD': {
            'PSNR': 0,
            'SSIM': 0,
            'TIME': 0,
            'FRAMES': 0,
            'COUNT': 0
        }
    }

    for bd, bi, gt in tqdm(loader):

        bd = bd.to(device)
        bi = bi.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_bi = model(bi)
            end_time = time.time()

        inference_time = end_time - start_time

        psnr_bi = metric_psnr(pred_bi, gt)
        ssim_bi = metric_ssim(pred_bi, gt)

        results['BI']['PSNR'] += psnr_bi
        results['BI']['SSIM'] += ssim_bi
        results['BI']['TIME'] += inference_time
        results['BI']['FRAMES'] += bi.shape[1]
        results['BI']['COUNT'] += 1

        with torch.no_grad():
            start_time = time.time()
            pred_bd = model(bd)
            end_time = time.time()

        inference_time = end_time - start_time

        psnr_bd = metric_psnr(pred_bd, gt)
        ssim_bd = metric_ssim(pred_bd, gt)

        results['BD']['PSNR'] += psnr_bd
        results['BD']['SSIM'] += ssim_bd
        results['BD']['TIME'] += inference_time
        results['BD']['FRAMES'] += bd.shape[1]
        results['BD']['COUNT'] += 1

    final_results = {}

    for key in ['BI', 'BD']:

        avg_psnr = results[key]['PSNR'] / results[key]['COUNT']
        avg_ssim = results[key]['SSIM'] / results[key]['COUNT']

        fps = results[key]['FRAMES'] / results[key]['TIME']

        latency = (results[key]['TIME'] / results[key]['FRAMES']) * 1000

        final_results[key] = {'PSNR': avg_psnr, 'SSIM': avg_ssim, 'FPS': fps, 'Latency': latency}

    return final_results

def main():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    loader = get_loader(batch_size=1, num_workers=4, pin_memory=True)
    models = {
        'BasicVSR': BasicVSRWrapper(checkpoint_path='./checkpoints/vsrnet_checkpoints/basicvsr_reds4_20120409-0e599677.pth', device=device),
        'EDVR': EDVRWrapper(checkpoint_path='./checkpoints/edvr_checkpoints/edvrl_c128b40_8x8_lr2e-4_600k_reds4_20220104-4509865f.pth', device=device),
        'RSDN': RSDNWrapper(checkpoint_path='./checkpoints/rsdn_checkpoints/RSDN.pth', device=device),
        'OVSR': OVSRWrapper(checkpoint_path='./checkpoints/ovsr_checkpoints/0721.pth', device=device)
    }

    for name, model in models.items():

        print(f"\n{'=' * 60}")
        print(f"Evaluating {name}")
        print(f"{'=' * 60}")

        results = evaluate(model=model, loader=loader, device=device)


        print("\n[BI Dataset]")
        print(f"PSNR               : {results['BI']['PSNR']:.4f}")
        print(f"SSIM               : {results['BI']['SSIM']:.4f}")
        print(f"FPS                : {results['BI']['FPS']:.2f}")
        print(f"Latency per frame  : {results['BI']['Latency']:.2f} ms")


        print("\n[BD Dataset]")
        print(f"PSNR               : {results['BD']['PSNR']:.4f}")
        print(f"SSIM               : {results['BD']['SSIM']:.4f}")
        print(f"FPS                : {results['BD']['FPS']:.2f}")
        print(f"Latency per frame  : {results['BD']['Latency']:.2f} ms")


if __name__ == "__main__":
    main()