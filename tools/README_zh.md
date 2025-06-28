# Tools 使用说明

[English](README.md) | **中文**

本目录包含了Depth Anything AC项目的工具脚本。

## 推理脚本 (infer.py)

### 基本使用

#### 单张图像推理
```bash
# 基本用法
python tools/infer.py --input image.jpg --output depth.png

# 指定模型和编码器
python tools/infer.py --input image.jpg --output depth.png --model checkpoints/depth_anything_AC_vits.pth --encoder vits

# 使用不同的颜色映射
python tools/infer.py --input image.jpg --output depth.png --colormap spectral
```

#### 批量图像推理
```bash
# 处理整个目录
python tools/infer.py --input images/ --output output/

# 批量处理并指定颜色映射
python tools/infer.py --input dataset/images/ --output results/ --colormap inferno
```

### 参数说明

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | `-i` | str | - | 输入图像路径或目录 (必需) |
| `--output` | `-o` | str | - | 输出路径 (必需) |
| `--model` | `-m` | str | `checkpoints/depth_anything_AC_vits.pth` | 模型权重路径 |
| `--encoder` | - | str | `vits` | 编码器类型 (`vits`, `vitb`, `vitl`) |
| `--colormap` | - | str | `inferno` | 颜色映射 (`inferno`, `spectral`, `gray`) |


### 使用示例

#### 示例1: 处理单张图像
```bash
python tools/infer.py -i test_image.jpg -o result.png
```
输出：
- `result.png` - 彩色深度图
- `result_raw.npy` - 原始深度数据

#### 示例2: 批量处理
```bash
python tools/infer.py -i photos/ -o depth_results/
```
对于 `photos/` 目录中的每张图像，会在 `depth_results/` 目录生成对应的深度图。


## 训练脚本

### train.sh
用于启动分布式训练的脚本。

```bash
bash tools/train.sh [GPU数量] [端口号]
```

### val.sh  
用于启动分布式评估的脚本。

```bash
bash tools/val.sh [GPU数量] [端口号]
```

---

更多详细信息请参考项目主[README](../README_zh.md)文档。 