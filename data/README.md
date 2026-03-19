# Dataset Preparation

This project uses two public datasets. We do **not** redistribute any
video data. Please download them directly from the official sources below.

---

## 1. Animal Kingdom

- **Official page**: https://sutdcv.github.io/Animal-Kingdom
- **Download**: Follow the instructions on the official page to request access.
- **Our subset**: We extracted videos of 6 wild feline species
  (lion, leopard, cheetah, ocelot, tiger, mountain lion) covering
  4 behavior categories: Walking, Running, Attacking, Keeping still.

---

## 2. MammalNet

- **Official page**: https://github.com/Vision-CAIR/MammalNet
- **Download**: Follow the repository instructions.
- **Our subset**: We extracted videos of 8 wild feline species
  (lion, tiger, jaguar, leopard, cheetah, lynx, clouded leopard,
  mountain lion) covering 6 behavior categories:
  Hunting, Mating, Fighting, Sleeping, Drinking, Eating.

---

## Subset Filtering Criteria

1. Only feline-category clips were retained from each dataset.
2. Clips with severe motion blur, heavy occlusion, or unidentifiable
   subjects were discarded.
3. Longer clips were trimmed into shorter segments to increase sample
   diversity.
4. Final splits: training / validation ≈ 8 : 2.

The data screening scripts used to apply these criteria are provided
in this repository (`models/` directory) and can be used to reproduce
our exact subsets from the original downloaded data.
