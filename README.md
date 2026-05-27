# Video Super-Resolution Benchmark

A PyTorch-based benchmark framework for evaluating multiple Video Super-Resolution (VSR) models on the Vid4 dataset.

Implemented models:

* [BasicVSR](https://arxiv.org/abs/2012.02181)
* [EDVR](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.pdf)
* [RSDN](https://arxiv.org/pdf/2008.00455)
* [OVSR](https://openaccess.thecvf.com/content/ICCV2021/papers/Yi_Omniscient_Video_Super-Resolution_ICCV_2021_paper.pdf)

The framework evaluates reconstruction quality and inference efficiency using:

* PSNR
* SSIM
* FPS
* Latency per frame

---

# Project Structure

```text
project/
├── checkpoints/
│   ├── basicvsr_checkpoints/
│   ├── edvr_checkpoints/
│   ├── rsdn_checkpoints/
│   └── ovsr_checkpoints/
│
├── data/
│   └── Vid4/
│       ├── BDx4/
│       ├── BIx4/
│       └── GT/
│
├── data.py
├── metrics.py
├── model.py
├── evaluate.py
├── rsdn.py
├── ovsr.py
└── README.md
```

---

# Installation

Install [Annaconda](https://www.anaconda.com/docs/getting-started/anaconda/install/overview) and Python3 on Ubuntu

Please strictly follow these installation steps:

Clone and navigate the repository
```bash
git clone https://github.com/vuminhkh0a/Video-Super-Resolution.git
cd Video-Super-Resolution
```

Create conda environment

```bash
conda create -n venv python=3.8 -y
conda activate venv
conda install pytorch=1.10.0 torchvision=0.11.0 cudatoolkit=11.3 -c pytorch -y
```

Install OpenMIM, MMCV, MMEngine

```bash
pip install -U openmim
mim install "mmcv==2.0.0"
```

Install mmagic and dependencies
```bash
git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip install -e . -v
cd ..
pip install -r requirements.txt
conda install ffmpeg -y
```
---

# Dataset Preparation

Download the [Vid4 dataset](https://www.kaggle.com/datasets/uom200647r/vid4-dataset?select=BIx4) (if the `data/` folder is empty) and organize it as:

```text
data/
└── Vid4/
    ├── BDx4/
    │   ├── calendar/
    │   ├── city/
    │   ├── foliage/
    │   └── walk/
    │
    ├── BIx4/
    │   ├── calendar/
    │   ├── city/
    │   ├── foliage/
    │   └── walk/
    │
    └── GT/
        ├── calendar/
        ├── city/
        ├── foliage/
        └── walk/
```

Each sequence folder contains multiple consecutive frames.

---

# Checkpoint Preparation

Place pretrained checkpoints inside (if the `checkpoints/` folder is empty):

```text
checkpoints/
├── basicvsr_checkpoints/
├── edvr_checkpoints/
├── rsdn_checkpoints/
└── ovsr_checkpoints/
```


---



# Run Evaluation

Before running, make sure:
- CUDA and GPU drivers are installed correctly
- PyTorch detects CUDA successfully

Quick CUDA check:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, you can run:

```bash
python3 main.py
```

# Live Demo

Before running, make sure:
- CUDA and GPU drivers are installed correctly
- PyTorch detects CUDA successfully

Quick CUDA check:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

If the output is `True`, run:

```bash
python3 demo_app.py
```

- Then navigate to the Gradio local URL or public URL shown in the terminal.
- If the local live demo system is failed, please navigate the [Backup Colab Live Demo System](https://colab.research.google.com/drive/1aZAdDl2wsIR6QAOqesv7VFRWrgG0UmbC?usp=sharing) (No BasicVSR and EDVR available on backup demo)
---

