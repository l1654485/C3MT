# C3MT

This repository is the official implementation of [C3MT: Confidence-Calibrated Contrastive Mean Teacher for Semi-supervised Medical Image Segmentation]. 

# Requirements:
- Python 3.10.16
- torch 2.0.0
- torchvision 0.15.0
- opencv-python 4.12.0.88
- numpy 1.26.4
- h5py 3.14.0
- scipy 1.15.3


# Datasets
**Dataset I**
ACDC. We use the code and preprocessed data by [CV-SSL-MIS](https://github.com/ziyangwang007/CV-SSL-MIS). 

**Dataset II**
Synapse. Following [DHC](https://github.com/xmed-lab/DHC), 20 samples were split for training, 4 samples for validation, and 6 samples for testing. We use the processed data by [MagicNet](https://github.com/DeepMed-Lab-ECNU/MagicNet).

**Dataset III**
LA. We use the code and preprocessed data by [BCP](https://github.com/DeepMed-Lab-ECNU/BCP). 

# Running
```
cd code
CUDA_VISIBLE_DEVICES=0 python C3MT.py --root_path ../data/ACDC --exp ACDC/C3MT --max_iterations 30000 --batch_size 16 --labeled_bs 8 --base_lr 0.01 --num_classes 4 --labeled_num 14
```

## Reference
* [MagicNet](https://github.com/DeepMed-Lab-ECNU/MagicNet)
* [CV-SSL-MIS](https://github.com/ziyangwang007/CV-SSL-MIS)
* [UCMT](https://github.com/Senyh/UCMT)

