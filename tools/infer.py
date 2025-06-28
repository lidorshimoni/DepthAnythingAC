import argparse
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from depth_anything.dpt import DepthAnything_AC


def normalize_depth(disparity_tensor):
    """
    Convert disparity to depth using normalization method.
    
    Args:
        disparity_tensor (torch.Tensor): Predicted disparity values
        
    Returns:
        torch.Tensor: depth values
    """
    eps = 1e-6
    
    disparity_min = disparity_tensor.min()
    disparity_max = disparity_tensor.max()
    normalized_disparity = (disparity_tensor - disparity_min) / (disparity_max - disparity_min + eps)
    
    
    return normalized_disparity


def load_model(model_path, encoder='vits'):
    """Load trained depth estimation model"""
    print(f"Loading model: {model_path}")
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'version': 'v2'},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'version': 'v2'},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'version': 'v2'}
    }
 
    model = DepthAnything_AC(model_configs[encoder])
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    
    return model


def preprocess_image(image_path, target_size=518):
    """Preprocess input image"""
    
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        raise ValueError(f"Cannot read image: {image_path}") 
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]
    scale = target_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    new_h = ((new_h + 13) // 14) * 14
    new_w = ((new_w + 13) // 14) * 14
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    image = image.unsqueeze(0)  
    
    return image, (h, w)


def postprocess_depth(depth_tensor, original_size):
    """Postprocess depth map"""
    print(f"Depth tensor shape: {depth_tensor.shape}, dimensions: {depth_tensor.dim()}")
    depth_tensor = depth_tensor.unsqueeze(1) 
    print(f"Processed tensor shape: {depth_tensor.shape}")
    h, w = original_size
    print(f"Target size: {h} x {w}")
    
    try:
        depth = F.interpolate(depth_tensor, size=(h, w), mode='bilinear', align_corners=True)
        depth = depth.squeeze().cpu().numpy()
        print(f"Final depth map shape: {depth.shape}")
        return depth
    except Exception as e:
        print(f"Interpolation failed: {str(e)}")
        return None


def save_depth_map(depth, output_path, colormap='inferno'):
    """Save depth map"""
    depth_raw_path = output_path.replace('.png', '_raw.npy')
    np.save(depth_raw_path, depth)

    if colormap == 'inferno':
        # Use OpenCV's INFERNO colormap
        depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    elif colormap == 'spectral':
        # Use matplotlib's Spectral colormap (reversed)
        spectral_cmap = cm.get_cmap('Spectral_r')  
        depth_colored = (spectral_cmap(depth) * 255).astype(np.uint8)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGBA2BGR)
    else:
        # Grayscale
        depth_colored = (depth * 255).astype(np.uint8)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, depth_colored)
    print(f"Depth map saved: {output_path}")
    print(f"Raw depth data saved: {depth_raw_path}")


def infer_single_image(model, image_path, output_path, colormap='inferno', depth_cap=80):
    """Perform depth estimation inference on a single image"""
    print(f"Processing: {image_path}")

    image_tensor, original_size = preprocess_image(image_path)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    with torch.no_grad():
        prediction = model(image_tensor)
        disparity_tensor = prediction['out']
        
        print(f"Model output shape (disparity): {disparity_tensor.shape}")
        print(f"Original image size: {original_size}")
        
        depth_tensor = normalize_depth(disparity_tensor)
        print(f"Converted to depth, shape: {depth_tensor.shape}")
    
    depth = postprocess_depth(depth_tensor, original_size)
            
    if depth is None:
        print("Reprocessing depth tensor...")
        if depth_tensor.dim() == 1:
            h, w = original_size
            expected_size = h * w
            if depth_tensor.shape[0] == expected_size:
                depth_tensor = depth_tensor.view(1, 1, h, w)
            else:
                import math
                side_length = int(math.sqrt(depth_tensor.shape[0]))
                if side_length * side_length == depth_tensor.shape[0]:
                    depth_tensor = depth_tensor.view(1, 1, side_length, side_length)
                else:
                    raise ValueError(f"Cannot reshape 1D tensor with size {depth_tensor.shape[0]}")
        depth = postprocess_depth(depth_tensor, original_size)
        if depth is None:
            raise ValueError("Cannot handle depth tensor shape problem")
    save_depth_map(depth, output_path, colormap)
    
    return depth


def infer_batch_images(model, input_dir, output_dir, colormap='inferno', depth_cap=80):
    """Batch process images"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    if not image_files:
        print(f"No image files found in directory {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(image_files)} images, starting batch processing...")
    for i, image_path in enumerate(image_files):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_depth.png")
        try:
            infer_single_image(model, image_path, output_path, colormap, depth_cap)
            print(f"Progress: {i+1}/{len(image_files)}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print("Batch processing completed!")


def main():
    parser = argparse.ArgumentParser(description='Depth Anything AC Depth Estimation Inference')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path (file or directory)')
    parser.add_argument('--model', '-m', type=str, 
                        default='checkpoints/depth_anything_v2_vits.pth',
                        help='Model weights path')
    parser.add_argument('--encoder', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl'],
                        help='Encoder type')
    parser.add_argument('--colormap', type=str, default='spectral',
                        choices=['inferno', 'spectral', 'gray'],
                        help='Depth map colormap')
    parser.add_argument('--depth_cap', type=float, default=80,
                        help='Maximum depth value for capping')
    args = parser.parse_args()
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    try:
        model = load_model(args.model, args.encoder)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    if os.path.isfile(args.input):
        if not os.path.splitext(args.output)[1]:
            args.output += '.png'
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        try:
            infer_single_image(model, args.input, args.output, args.colormap, args.depth_cap)
            print("Inference completed!")
        except Exception as e:
            print(f"Inference failed: {str(e)}")
    
    elif os.path.isdir(args.input):
        infer_batch_images(model, args.input, args.output, args.colormap, args.depth_cap)
    else:
        print(f"Error: Input path does not exist: {args.input}")


if __name__ == '__main__':
    main() 