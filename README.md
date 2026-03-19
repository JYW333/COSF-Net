# COSF-Net
---

## Overview

This repository contains the core source code for the paper listed above.

Wild feline behavior recognition in field settings is challenged by
camouflage backgrounds, large scale variations across species, occlusion,
and low-light conditions. We propose **COSF-Net**, a Collaborative SlowFast
Network that integrates a **Heterogeneous Attention Feature Fusion (HAFF)**
module composed of two novel 3D attention sub-modules:

| Module | Role in HAFF |
|--------|-------------|
| **MS-FCA-3D** (Multi-Scale Fine-grained Channel Attention 3D) | Fast→Slow direction: multi-scale spatial pyramid + gated residual fusion for scale adaptation and noise suppression |
| **DC-BiFormer-3D** (Dynamic Context Bi-level Routing Attention 3D) | Slow→Fast direction: dynamic window predictor + spatio-temporal context fusion for adaptive receptive field and global context compensation |

### Main Results

| Dataset | Top-1 Acc | Params | FLOPs |
|---------|-----------|--------|-------|
| Animal Kingdom (feline subset) | **83.12%** | lower than baselines | lower than baselines |
| MammalNet (feline subset) | **64.30%** | lower than baselines | lower than baselines |

---

## Repository Structure
```
COSF-Net/
├── models/
│   ├── custom_attention_3d.py        # MS-FCA-3D and DC-BiFormer-3D
│   └── custom_video_model_builder.py # Full COSF-Net (SlowFastDualAttention)
├── data/
│   ├── README.md                     # Dataset download and preparation guide
│   ├── animal_kingdom/
│   │   ├── filter_feline_subset.py   # Feline subset extraction
│   │   ├── analyze_video_enhanced.py # Duration and statistics analysis
│   │   └── Video Dataset Analyzer.py # Local video verification
│   └── mammalnet/
│       ├── analyze_big_cats_enhanced_v2.py  # Subset extraction and stats
│       └── count.py                  # Final split verification
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dependencies

**Python**: 3.8  
**CUDA**: 11.3

Install PyTorch first (must match your CUDA version):
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install remaining dependencies:
```bash
pip install -r requirements.txt
```

Note: `torch` and `torchvision` in `requirements.txt` are listed for
reference. Install them via the command above to ensure CUDA compatibility.

---

## Quick Start
```python
import torch
from models.custom_attention_3d import MultiScaleFCALayer3D, DynamicContextBiFormer3D

# MS-FCA-3D
x = torch.randn(2, 64, 8, 56, 56)   # [B, C, T, H, W]
ms_fca = MultiScaleFCALayer3D(channel=64, num_scales=3)
out = ms_fca(x)   # shape: [2, 64, 8, 56, 56]

# DC-BiFormer-3D
dc_bi = DynamicContextBiFormer3D(dim=64, num_heads=8, base_win=7, topk=4)
out = dc_bi(x)    # shape: [2, 64, 8, 56, 56]
```

---

## Dataset

Please refer to [`data/README.md`](data/README.md) for instructions on
downloading Animal Kingdom and MammalNet and reproducing our feline subsets.

---

## Citation

If you use this code, please cite our paper:
```bibtex
@article{ji2026cosfnet,
  title   = {Enhanced Wild Feline Behavior Recognition via Collaborative
             SlowFast Network with Heterogeneous Attention Fusion},
  author  = {Ji, Yawei and Zhao, Yaqin and Lu, Xu and Sun, Xiangna},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

This project is licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
Free for academic and non-commercial use with attribution.  
Commercial use is **prohibited** without explicit written permission.

© 2026 Yawei Ji, Yaqin Zhao, Xu Lu, Xiangna Sun — Nanjing Forestry University