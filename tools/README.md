# Tools Usage Guide

**English** | [中文](README_zh.md)

This directory contains tool scripts for the Depth Anything AC project.

## Inference Script (infer.py)
### Basic Usage

#### Single Image Inference
```bash
# Basic usage
python tools/infer.py --input image.jpg --output depth.png

# Specify model and encoder
python tools/infer.py --input image.jpg --output depth.png --model checkpoints/depth_anything_AC_vits.pth --encoder vits

# Use different colormap
python tools/infer.py --input image.jpg --output depth.png --colormap spectral
```

#### Batch Image Inference
```bash
# Process entire directory
python tools/infer.py --input images/ --output output/

# Batch processing with specified colormap
python tools/infer.py --input dataset/images/ --output results/ --colormap inferno
```

### Parameters

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | str | - | Input image path or directory (required) |
| `--output` | `-o` | str | - | Output path (required) |
| `--model` | `-m` | str | `checkpoints/depth_anything_AC_vits.pth` | Model weight path |
| `--encoder` | - | str | `vits` | Encoder type (`vits`, `vitb`, `vitl`) |
| `--colormap` | - | str | `inferno` | Colormap (`inferno`, `spectral`, `gray`) |

### Usage Examples

#### Example 1: Process Single Image
```bash
python tools/infer.py -i test_image.jpg -o result.png
```
Output:
- `result.png` - Colored depth map
- `result_raw.npy` - Raw depth data

#### Example 2: Batch Processing
```bash
python tools/infer.py -i photos/ -o depth_results/
```
For each image in the `photos/` directory, corresponding depth maps will be generated in the `depth_results/` directory.


## Training Scripts

### train.sh
Script for launching distributed training.

```bash
bash tools/train.sh [NUM_GPUS] [PORT]
```

### val.sh  
Script for launching distributed evaluation.

```bash
bash tools/val.sh [NUM_GPUS] [PORT]
```

---

For more detailed information, please refer to the main project  [README](../README.md) documentation. 