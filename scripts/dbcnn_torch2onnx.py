"""
Convert DBCNN (Double-blind CNN) PyTorch model to ONNX format.

DBCNN uses VGG16 + SCNN with bilinear pooling. The ONNX model accepts
variable-resolution input (no fixed crop required).

Usage:
    python scripts/dbcnn_torch2onnx.py \
        --output_dir /mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add IQA-PyTorch to path
IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

from pyiqa.archs.dbcnn_arch import DBCNN
from pyiqa.archs.arch_util import load_pretrained_network


class DBCNNWrapper(nn.Module):
    """Wrapper that runs DBCNN in eval mode with a fixed resolution input.

    The original DBCNN.forward() includes preprocessing (normalization) and
    handles variable-size inputs via adaptive average pooling internally.
    This wrapper skips the built-in preprocessing so that normalization is
    done in Java, matching the Java adapter's expectations.
    """

    def __init__(self, model):
        super().__init__()
        self.features1 = model.features1  # VGG16
        self.features2 = model.features2  # SCNN
        self.fc = model.fc

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) — already ImageNet-normalised by the Java side.
        Returns:
            score: (B, 1)
        """
        f1 = self.features1(x)  # (B, 512, h1, w1)
        f2 = self.features2(x)  # (B, 128, h2, w2)

        # Match spatial sizes
        if f1.shape[2:] != f2.shape[2:]:
            f2 = nn.functional.adaptive_avg_pool2d(f2, f1.shape[2:])

        # Bilinear pooling
        B, C1, H, W = f1.shape
        C2 = f2.shape[1]
        f1_flat = f1.view(B, C1, H * W)       # (B, 512, N)
        f2_flat = f2.view(B, C2, H * W)       # (B, 128, N)
        bi = torch.bmm(f1_flat, f2_flat.transpose(1, 2))  # (B, 512, 128)
        bi = bi.view(B, C1 * C2) / (H * W)    # (B, 65536)

        # Signed sqrt + L2 norm
        bi = torch.sign(bi) * torch.sqrt(torch.abs(bi) + 1e-12)
        bi = nn.functional.normalize(bi, p=2, dim=1)

        score = self.fc(bi)  # (B, 1)
        return score


def main():
    parser = argparse.ArgumentParser(description='Convert DBCNN to ONNX')
    parser.add_argument('--pretrained', type=str, default='koniq',
                        help='Pretrained variant (koniq, live, livec, etc.)')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/',
                        help='Output directory for ONNX model')
    parser.add_argument('--height', type=int, default=384,
                        help='Export height (default 384)')
    parser.add_argument('--width', type=int, default=512,
                        help='Export width (default 512)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('DBCNN PyTorch → ONNX Conversion')
    print('=' * 60)

    # Build model
    print(f'Loading DBCNN (pretrained={args.pretrained})...')
    model = DBCNN(pretrained=args.pretrained)
    model.eval()

    wrapper = DBCNNWrapper(model)
    wrapper.eval()

    # Dummy input
    dummy = torch.randn(1, 3, args.height, args.width)

    output_path = os.path.join(args.output_dir, 'dbcnn_model.onnx')
    print(f'Exporting to {output_path}...')

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['score'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
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
    test_input = np.random.randn(1, 3, args.height, args.width).astype(np.float32)
    out = sess.run(None, {'input': test_input})
    print(f'Output shape: {out[0].shape}, value: {out[0].flatten()[0]:.4f}')
    print('Done.')


if __name__ == '__main__':
    main()
