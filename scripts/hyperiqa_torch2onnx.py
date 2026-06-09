"""
Convert HyperIQA (Hyper-Network) PyTorch model to ONNX format.

HyperIQA uses a ResNet50 backbone with a hyper-network that generates
FC weights dynamically. Only ``forward_patch`` (single 224x224 crop) is
exported; multi-crop averaging is handled in Java.

Usage:
    python scripts/hyperiqa_torch2onnx.py \
        --output_dir /mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

from pyiqa.archs.hypernet_arch import HyperNet


class HyperIQAWrapper(nn.Module):
    """Thin wrapper exposing ``forward_patch`` for ONNX export.

    The full HyperNet.forward() calls uniform_crop internally; we bypass
    that because Java handles cropping.  We also skip the built-in
    ``preprocess()`` since ImageNet normalization is done in Java.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) — already ImageNet-normalised.
        Returns:
            score: (B, 1)
        """
        return self.model.forward_patch(x)


def main():
    parser = argparse.ArgumentParser(description='Convert HyperIQA to ONNX')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('HyperIQA PyTorch → ONNX Conversion')
    print('=' * 60)

    print('Loading HyperNet (resnet50-koniq)...')
    model = HyperNet(pretrained='resnet50-koniq')
    model.eval()

    wrapper = HyperIQAWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)

    output_path = os.path.join(args.output_dir, 'hyperiqa_model.onnx')
    print(f'Exporting to {output_path}...')

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['score'],
        dynamic_axes={
            'input': {0: 'batch'},
            'score': {0: 'batch'},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f'Exported: {output_path}')

    # Verify
    print('Verifying with ONNX Runtime...')
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = sess.run(None, {'input': test_input})
    print(f'Output shape: {out[0].shape}, value: {out[0].flatten()[0]:.4f}')
    print('Done.')


if __name__ == '__main__':
    main()
