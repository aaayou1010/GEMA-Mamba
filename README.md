# GEMA-Mamba

The official implementation of GEMA-Mamba.

## Overview

This repository contains the training, evaluation, visualization, and data preparation code for GEMA-Mamba.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This project includes scripts for multiple medical image segmentation datasets, including:

- ISIC 2017
- ISIC 2018
- ACDC
- LIDC

Useful preparation scripts:

- `prepare_isic2018.py`
- `prepare_lidc_from_dicom_xml.py`
- `convert_acdc_patient_h5_to_npz.py`

## Training

Run training with:

```bash
python train.py --datasets_name <dataset_name> --epochs 1000 --batch_size 36 --work_dir <output_dir>
```

Example:

```bash
python train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir ./outputs/isic2018
```

## Evaluation

Run evaluation with:

```bash
python test.py --datasets_name <dataset_name> --batch_size 36 --work_dir <output_dir> --best_model_path <checkpoint_path>
```

## Inference and Export

Useful scripts for inference and result export:

- `infer_lidc.py`
- `export_testset_itksnap.py`
- `test_Monu.py`

## Project Structure

```text
models/GEMAMamba/     model definitions
train.py              training entry
test.py               evaluation entry
gema_engine.py        training and validation loops
gema_utils.py         loss, metrics, visualization, and helpers
gema_gradcam.py       Grad-CAM utilities
```

## Acknowledgement

This project is built with inspiration from:

- [VMamba](https://github.com/MzeroMiko/VMamba)
- [VM-UNet](https://github.com/JCruan519/VM-UNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
