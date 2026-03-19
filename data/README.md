# Dataset Preparation

This project uses two public datasets. We do **not** redistribute any
video data. Please download them directly from the official sources below.

---

## 1. Animal Kingdom

- **Official page**: https://sutdcv.github.io/Animal-Kingdom
- **Download**: Follow the instructions on the official page to request
  access to the video clips.
- **Our subset**: We extracted videos of 6 wild feline species
  (lion, leopard, cheetah, ocelot, tiger, mountain lion) covering
  4 behavior categories: Walking, Running, Attacking, Keeping still.

### Provided scripts (`data/animal_kingdom/`)

| Script | Purpose |
|--------|---------|
| `filter_feline_subset.py` | Filters feline-category clips from the full Animal Kingdom annotation file and organises them by behavior class |
| `analyze_video_enhanced.py` | Parses video duration metadata and computes per-class statistics |
| `Video Dataset Analyzer.py` | Scans local video files to verify frame count, resolution, and duration after download |

---

## 2. MammalNet

- **Official page**: https://github.com/Vision-CAIR/MammalNet
- **Download**: Follow the repository instructions.
- **Our subset**: We extracted videos of 8 wild feline species
  (lion, tiger, jaguar, leopard, cheetah, lynx, clouded leopard,
  mountain lion) covering 6 behavior categories:
  Hunting, Mating, Fighting, Sleeping, Drinking, Eating.

### Provided scripts (`data/mammalnet/`)

| Script | Purpose |
|--------|---------|
| `analyze_big_cats_enhanced_v2.py` | Filters big-cat clips from MammalNet annotations and computes subset statistics |
| `count.py` | Counts final video numbers per species and behavior category, and verifies the 8:2 train/validation split |

---

## Subset Filtering Criteria

1. Only feline-category clips were retained from each dataset.
2. Clips with severe motion blur, heavy occlusion, or unidentifiable
   subjects were discarded.
3. Longer clips (MammalNet) were trimmed into shorter segments to
   increase sample diversity.
4. Final splits: training / validation ≈ 8 : 2.
   All reported metrics are evaluated on the validation set.

---

## Reproducing Our Subsets

After downloading the original datasets, run the scripts in order:
```bash
# Animal Kingdom
python data/animal_kingdom/filter_feline_subset.py
python data/animal_kingdom/analyze_video_enhanced.py

# MammalNet
python data/mammalnet/analyze_big_cats_enhanced_v2.py
python data/mammalnet/count.py
```