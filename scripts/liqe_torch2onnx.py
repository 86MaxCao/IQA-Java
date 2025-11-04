"""
Convert LIQE PyTorch models to ONNX format

This script converts:
1. CLIP model (ViT-B-32.pt) -> clip_model.onnx
2. LIQE model (liqe_mix.pt) -> liqe_model.onnx (注意：由于LIQE模型包含动态操作，需要特殊处理)
3. Text features (liqe_text_feat_mix.pt) -> 保存为 numpy 格式供 Java 使用

注意：LIQE 模型的 forward 包含动态操作（unfold, 动态 patch 选择），
需要创建一个简化的 wrapper 来导出 ONNX。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path

# Add current directory to path to import udtf_liqe
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import from udtf_liqe
from udtf_liqe import (
    LIQE, load_clip_model, load_pretrained_network,
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD,
    qualitys, scenes, dists_map
)


class CLIPImageEncoderWrapper(nn.Module):
    """Wrapper for CLIP image encoder to export to ONNX"""
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        
    def forward(self, x):
        """
        Encode image to features
        Input: (batch_size, 3, 224, 224)
        Output: (batch_size, feature_dim)
        """
        try:
            image_features = self.clip_model.encode_image(x, pos_embedding=True)
        except TypeError:
            image_features = self.clip_model.encode_image(x)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features


class LIQEWrapper(nn.Module):
    """Simplified wrapper for LIQE model to export to ONNX
    
    This wrapper handles the patch extraction and selection logic
    that can be done in Java preprocessing.
    """
    def __init__(self, liqe_model, text_features):
        super().__init__()
        self.liqe_model = liqe_model
        self.text_features = nn.Parameter(text_features, requires_grad=False)
        
        # Store constants
        self.default_mean = nn.Parameter(
            torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1),
            requires_grad=False
        )
        self.default_std = nn.Parameter(
            torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1),
            requires_grad=False
        )
        
    def forward(self, image_patches, logit_scale):
        """
        Forward pass for LIQE model
        
        Args:
            image_patches: (batch_size, num_patches, 3, 224, 224) - pre-extracted patches
            logit_scale: (1,) - temperature scale from CLIP model
        
        Returns:
            quality_score: (batch_size,) - quality score in range [1, 5]
        """
        batch_size = image_patches.size(0)
        num_patches = image_patches.size(1)
        
        # Reshape for batch processing: (batch_size * num_patches, 3, 224, 224)
        image_patches_flat = image_patches.view(batch_size * num_patches, 3, 224, 224)
        
        # Normalize patches
        image_patches_flat = (image_patches_flat - self.default_mean) / self.default_std
        
        # Encode images using CLIP
        try:
            image_features = self.liqe_model.clip_model.encode_image(
                image_patches_flat, pos_embedding=True
            )
        except TypeError:
            image_features = self.liqe_model.clip_model.encode_image(image_patches_flat)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Compute similarity with text features
        logit_scale_exp = logit_scale.exp()
        logits_per_image = logit_scale_exp * image_features @ self.text_features.t()
        
        # Reshape back: (batch_size, num_patches, num_text_features)
        logits_per_image = logits_per_image.view(batch_size, num_patches, -1)
        
        # Average over patches
        logits_per_image = logits_per_image.mean(1)  # (batch_size, num_text_features)
        
        # Softmax
        logits_per_image = F.softmax(logits_per_image, dim=1)
        
        # Reshape for MTL: (batch_size, len(qualitys), len(scenes), len(dists_map))
        if self.liqe_model.mtl:
            logits_per_image = logits_per_image.view(
                batch_size, len(qualitys), len(scenes), len(dists_map)
            )
            logits_quality = logits_per_image.sum(3).sum(2)  # (batch_size, len(qualitys))
        else:
            logits_quality = logits_per_image.view(batch_size, len(qualitys))
        
        # Compute quality score: weighted sum
        quality = (
            1 * logits_quality[:, 0] + 
            2 * logits_quality[:, 1] + 
            3 * logits_quality[:, 2] + 
            4 * logits_quality[:, 3] + 
            5 * logits_quality[:, 4]
        )
        
        return quality


def export_clip_model(clip_model_path, output_path, device='cpu'):
    """Export CLIP image encoder to ONNX"""
    print(f"Loading CLIP model from {clip_model_path}...")
    
    # Load CLIP model
    clip_model = load_clip_model(
        backbone='ViT-B/32',
        device=device,
        model_path=clip_model_path
    )
    clip_model.eval()
    
    # Create wrapper
    wrapper = CLIPImageEncoderWrapper(clip_model)
    wrapper.eval()
    
    # Create dummy input: (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting CLIP model to {output_path}...")
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['image_features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'image_features': {0: 'batch_size'}
        },
        opset_version=14,  # Use opset 11 for better compatibility
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ CLIP model exported to {output_path}")
    
    # Test the exported model
    print("Testing exported CLIP model...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession(output_path)
    
    # Test with dummy input
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = ort_session.run(None, {'image': test_input})
    print(f"✓ ONNX model output shape: {outputs[0].shape}")
    
    return wrapper


def export_liqe_model(
    liqe_model_path,
    clip_model_path,
    text_feat_path,
    output_path,
    device='cpu',
    num_patches=15
):
    """Export LIQE model to ONNX (with simplified wrapper)"""
    print(f"Loading LIQE model from {liqe_model_path}...")
    
    # Load LIQE model
    liqe_model = LIQE(
        pretrained='mix',
        pretrained_model_path=liqe_model_path,
        clip_model_path=clip_model_path,
        text_feat_cache_path=text_feat_path
    ).eval()
    
    # Load text features
    text_features = torch.load(text_feat_path, map_location='cpu')
    
    # Create wrapper
    wrapper = LIQEWrapper(liqe_model, text_features)
    wrapper.eval()
    
    # Create dummy inputs
    # Input: (batch_size=1, num_patches, 3, 224, 224)
    dummy_patches = torch.randn(1, num_patches, 3, 224, 224)
    # Logit scale from CLIP model
    dummy_logit_scale = liqe_model.clip_model.logit_scale
    
    print(f"Exporting LIQE model to {output_path}...")
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_patches, dummy_logit_scale),
        output_path,
        input_names=['image_patches', 'logit_scale'],
        output_names=['quality_score'],
        dynamic_axes={
            'image_patches': {0: 'batch_size'},
            'quality_score': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✓ LIQE model exported to {output_path}")
    
    # Test the exported model
    print("Testing exported LIQE model...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession(output_path)
    
    # Test with dummy input
    test_patches = np.random.randn(1, num_patches, 3, 224, 224).astype(np.float32)
    test_logit_scale = np.array([dummy_logit_scale.item()], dtype=np.float32)
    outputs = ort_session.run(
        None,
        {
            'image_patches': test_patches,
            'logit_scale': test_logit_scale
        }
    )
    print(f"✓ ONNX model output shape: {outputs[0].shape}")
    print(f"✓ Quality score range: [{outputs[0].min():.2f}, {outputs[0].max():.2f}]")
    
    return wrapper


def save_text_features(text_feat_path, output_dir):
    """Save text features as JSON format for Java use
    
    JSON format is easy to read and parse in Java using Jackson/Gson.
    """
    import json
    
    print(f"Loading text features from {text_feat_path}...")
    text_features = torch.load(text_feat_path, map_location='cpu')
    
    # Convert to numpy
    if isinstance(text_features, torch.Tensor):
        # Detach if requires grad, then convert to numpy
        if text_features.requires_grad:
            text_features_np = text_features.detach().numpy()
        else:
            text_features_np = text_features.numpy()
    else:
        text_features_np = np.array(text_features)
    
    # Ensure float32 for consistency
    if text_features_np.dtype != np.float32:
        text_features_np = text_features_np.astype(np.float32)
    
    shape = text_features_np.shape
    print(f"  Shape: {shape}, Dtype: {text_features_np.dtype}")
    
    # Save as JSON (easiest for Java to read)
    json_path = os.path.join(output_dir, 'text_features.json')
    print(f"Saving as JSON to {json_path}...")
    # Convert to nested list for JSON
    text_features_list = text_features_np.tolist()
    json_data = {
        'shape': list(shape),
        'dtype': 'float32',
        'data': text_features_list
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    
    file_size_mb = os.path.getsize(json_path) / 1024 / 1024
    print(f"✓ Text features saved to {json_path} ({file_size_mb:.2f} MB)")
    print(f"  Shape: {shape}, Dtype: {text_features_np.dtype}")
    
    return text_features_np


def main():
    parser = argparse.ArgumentParser(description='Convert LIQE models to ONNX')
    parser.add_argument(
        '--liqe_model_path',
        type=str,
        default='weights/liqe_mix.pt',
        help='Path to LIQE model weights'
    )
    parser.add_argument(
        '--clip_model_path',
        type=str,
        default='weights/ViT-B-32.pt',
        help='Path to CLIP model weights'
    )
    parser.add_argument(
        '--text_feat_path',
        type=str,
        default='weights/liqe_text_feat_mix.pt',
        help='Path to text features file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='weights/onnx',
        help='Output directory for ONNX models'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for conversion'
    )
    parser.add_argument(
        '--num_patches',
        type=int,
        default=15,
        help='Number of patches for LIQE model'
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Resolve paths
    if not os.path.isabs(args.liqe_model_path):
        liqe_model_path = os.path.join(project_root, args.liqe_model_path)
    else:
        liqe_model_path = args.liqe_model_path
    
    if not os.path.isabs(args.clip_model_path):
        clip_model_path = os.path.join(project_root, args.clip_model_path)
    else:
        clip_model_path = args.clip_model_path
    
    if not os.path.isabs(args.text_feat_path):
        text_feat_path = os.path.join(project_root, args.text_feat_path)
    else:
        text_feat_path = args.text_feat_path
    
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(project_root, args.output_dir)
    else:
        output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("LIQE PyTorch to ONNX Conversion")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    
    # 1. Export CLIP model
    print("\n[1/3] Exporting CLIP model...")
    clip_output = os.path.join(output_dir, 'clip_model.onnx')
    try:
        export_clip_model(clip_model_path, clip_output, args.device)
    except Exception as e:
        print(f"✗ Error exporting CLIP model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Export LIQE model
    print("\n[2/3] Exporting LIQE model...")
    liqe_output = os.path.join(output_dir, 'liqe_model.onnx')
    try:
        export_liqe_model(
            liqe_model_path,
            clip_model_path,
            text_feat_path,
            liqe_output,
            args.device,
            args.num_patches
        )
    except Exception as e:
        print(f"✗ Error exporting LIQE model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Save text features
    print("\n[3/3] Saving text features...")
    try:
        save_text_features(text_feat_path, output_dir)
    except Exception as e:
        print(f"✗ Error saving text features: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {clip_output}")
    print(f"  - {liqe_output}")
    print(f"  - {os.path.join(output_dir, 'text_features.json')}")
    print("\nThese files can now be used in Java with ONNX Runtime.")
    print("The text_features.json can be easily loaded using Jackson or Gson in Java.")


if __name__ == '__main__':
    main()

