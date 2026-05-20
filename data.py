import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os


def load_sequence(folder):
    '''
    Reads path of a folder that contains frames and returns tensor with the shape of (N_frames, C, H, W) 
    '''
    frames = []
    files = sorted(os.listdir(folder))
    
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0 
        frames.append(img)

    frames = np.stack(frames, axis=0) 
    frames = np.transpose(frames, (0, 3, 1, 2)) 

    return torch.from_numpy(frames)


class Vid4_Dataset(Dataset):
    '''
    Vid4 video super-resolution dataset.

    Each sample corresponds to a video sequence folder containing
    multiple consecutive frames.

    Inputs:
        bd_paths (list): List of paths to BDx4 sequences.
        bi_paths (list): List of paths to BIx4 sequences.
        gt_paths (list): List of paths to GT sequences.

    Outputs:
        tuple:
            bd (Tensor): BD input sequence with shape (N_frames, C, H, W)
            bi (Tensor): BI input sequence with shape (N_frames, C, H, W)
            gt (Tensor): Ground-truth sequence with shape (N_frames, C, H, W)
    '''

    def __init__(self, bd_paths, bi_paths, gt_paths):
        super().__init__()
        self.bd_paths = bd_paths
        self.bi_paths = bi_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.bd_paths)
    
    def __getitem__(self, i):
        return load_sequence(self.bd_paths[i]), load_sequence(self.bi_paths[i]), load_sequence(self.gt_paths[i])
    


def get_paths():
    '''
    Load and organize sequence folder paths from the Vid4 dataset.

    Dataset structure:
        ./data/Vid4/
            ├── BDx4/
            ├── BIx4/
            └── GT/

    Returns:
        tuple:
            BD_path_list (list): List of BDx4 sequence paths.
            BI_path_list (list): List of BIx4 sequence paths.
            GT_path_list (list): List of GT sequence paths.
    '''

    BD_path = './data/Vid4/BDx4'
    BI_path = './data/Vid4/BIx4'
    GT_path = './data/Vid4/GT'
    BD_path_list = []
    BI_path_list = []
    GT_path_list = []

    for bd_sample_folder, bi_sample_folder, gt_sample_folder in zip(sorted(os.listdir(BD_path)), sorted(os.listdir(BI_path)), sorted(os.listdir(GT_path))):
        bd_folder = os.path.join(BD_path, bd_sample_folder)
        bi_folder = os.path.join(BI_path, bi_sample_folder)
        gt_folder = os.path.join(GT_path, gt_sample_folder)
        BD_path_list.append(bd_folder)
        BI_path_list.append(bi_folder)
        GT_path_list.append(gt_folder)
    
    return BD_path_list, BI_path_list, GT_path_list

def get_dataset():
    BD_path_list, BI_path_list, GT_path_list = get_paths()
    return Vid4_Dataset(BD_path_list, BI_path_list, GT_path_list)

def get_loader(batch_size, num_workers, pin_memory):
    dataset = get_dataset()
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


