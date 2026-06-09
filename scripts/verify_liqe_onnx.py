"""
Verify LIQE ONNX export by comparing PyTorch vs ONNX outputs.

This script:
1. Loads the original PyTorch LIQE model
2. Loads the ONNX models (CLIP + LIQE)
3. Creates a deterministic test image
4. Compares PyTorch and ONNX outputs
5. Reports PASS/FAIL based on tolerance
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import json

# Constants
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
scenes = [
    'animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant',
    'still_life', 'others'
]
dists_map = [
    'jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color',
    'contrast', 'overexposure', 'underexposure', 'spatial', 'quantization',
    'other'
]

# Paths
LIQE_MODEL_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/liqe_mix.pt'
CLIP_MODEL_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/ViT-B-32.pt'
TEXT_FEAT_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/liqe_text_feat_mix.pt'
CLIP_ONNX_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/clip_model.onnx'
LIQE_ONNX_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/liqe_model.onnx'
TEXT_FEAT_JSON_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/text_features.json'

TOLERANCE = 0.05


def extract_patches(image_tensor, step=32, num_patch=15):
    """Extract patches from image tensor exactly as LIQE does.

    Args:
        image_tensor: (1, 3, H, W) - raw image tensor, NOT normalized
        step: sliding window step
        num_patch: number of patches to select

    Returns:
        patches: (1, num_patch, 3, 224, 224) - selected patches, NOT normalized
    """
    bs = image_tensor.size(0)
    # Extract patches using unfold (same as LIQE forward)
    x = image_tensor.unfold(2, 224, step).unfold(3, 224, step)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, 3, 224, 224)

    total_patches = x.size(1)
    if total_patches < num_patch:
        num_patch = total_patches

    # Deterministic patch selection (eval mode)
    sel_step = max(1, total_patches // num_patch)
    sel = torch.zeros(num_patch, dtype=torch.long)
    for i in range(num_patch):
        sel[i] = sel_step * i

    x = x[:, sel, ...]
    return x, num_patch


def run_pytorch(image_tensor):
    """Run the full PyTorch LIQE pipeline."""
    # Add IQA-PyTorch to path
    sys.path.insert(0, '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch')
    from pyiqa.archs.liqe_arch import LIQE

    print("Loading PyTorch LIQE model...")
    model = LIQE(
        pretrained='mix',
        pretrained_model_path=LIQE_MODEL_PATH,
    ).eval()

    print("Running PyTorch inference...")
    with torch.no_grad():
        # LIQE forward expects (N, 3, H, W) raw image
        # It normalizes internally
        score = model(image_tensor)

    return score.item()


def run_onnx(image_tensor):
    """Run the ONNX LIQE pipeline.

    The LIQEWrapper ONNX model takes:
      - image_patches: (batch, num_patches, 3, 224, 224) - NOT normalized
      - logit_scale: (1,) - scalar

    It normalizes internally, encodes via CLIP, computes similarity, and
    returns quality score.
    """
    print("Loading ONNX LIQE model...")
    liqe_session = ort.InferenceSession(
        LIQE_ONNX_PATH,
        providers=['CPUExecutionProvider']
    )

    # Get logit_scale from the PyTorch CLIP model
    # (needed as input to LIQE ONNX)
    docs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs'
    )
    sys.path.insert(0, docs_dir)
    from udtf_liqe import load_clip_model
    clip_model = load_clip_model(
        backbone='ViT-B/32', device='cpu', model_path=CLIP_MODEL_PATH
    )
    logit_scale = clip_model.logit_scale.detach().numpy().astype(np.float32)
    # Ensure it's a 1-D array with shape (1,)
    if logit_scale.ndim == 0:
        logit_scale = np.array([logit_scale.item()], dtype=np.float32)
    print(f"  logit_scale value: {logit_scale} (exp={np.exp(logit_scale)})")

    # Extract patches from image (NOT normalized)
    patches, num_patch = extract_patches(image_tensor, step=32, num_patch=15)
    print(f"  Extracted {num_patch} patches, shape: {patches.shape}")

    # Convert patches to numpy
    patches_np = patches.numpy().astype(np.float32)

    print("Running ONNX LIQE inference...")
    outputs = liqe_session.run(
        None,
        {
            'image_patches': patches_np,
            'logit_scale': logit_scale,
        }
    )

    score = outputs[0][0]
    return float(score)


def main():
    print("=" * 60)
    print("LIQE ONNX Verification: PyTorch vs ONNX")
    print("=" * 60)

    # Check files exist
    for path, name in [
        (CLIP_ONNX_PATH, "CLIP ONNX"),
        (LIQE_ONNX_PATH, "LIQE ONNX"),
        (TEXT_FEAT_JSON_PATH, "Text Features JSON"),
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  {name}: {path} ({size_mb:.2f} MB)")
        else:
            print(f"  ERROR: {name} not found at {path}")
            return

    # Create deterministic test image: (1, 3, 640, 360) simulating a real image
    print("\nCreating test image...")
    torch.manual_seed(42)
    np.random.seed(42)
    # Use a realistic-looking random image in [0, 1] range
    test_image = torch.rand(1, 3, 360, 640)
    print(f"  Test image shape: {test_image.shape}")
    print(f"  Test image range: [{test_image.min():.4f}, {test_image.max():.4f}]")

    # Run PyTorch
    print("\n--- PyTorch Pipeline ---")
    pytorch_score = run_pytorch(test_image.clone())
    print(f"  PyTorch score: {pytorch_score:.6f}")

    # Run ONNX
    print("\n--- ONNX Pipeline ---")
    onnx_score = run_onnx(test_image.clone())
    print(f"  ONNX score: {onnx_score:.6f}")

    # Compare
    abs_diff = abs(pytorch_score - onnx_score)
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  PyTorch score: {pytorch_score:.6f}")
    print(f"  ONNX score:    {onnx_score:.6f}")
    print(f"  Abs diff:      {abs_diff:.6f}")
    print(f"  Tolerance:     {TOLERANCE}")

    if abs_diff < TOLERANCE:
        print(f"\n  >>> PASS (diff {abs_diff:.6f} < {TOLERANCE}) <<<")
    else:
        print(f"\n  >>> FAIL (diff {abs_diff:.6f} >= {TOLERANCE}) <<<")

    # Print file info
    print("\nONNX file details:")
    for path in [CLIP_ONNX_PATH, LIQE_ONNX_PATH, TEXT_FEAT_JSON_PATH]:
        size_mb = os.path.getsize(path) / 1024 / 1024
        data_path = path + '.data'
        if os.path.exists(data_path):
            data_size_mb = os.path.getsize(data_path) / 1024 / 1024
            print(f"  {os.path.basename(path)}: {size_mb:.2f} MB "
                  f"(+ .data: {data_size_mb:.2f} MB)")
        else:
            print(f"  {os.path.basename(path)}: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
