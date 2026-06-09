"""
Convert TReS PyTorch model to ONNX format.

TReS uses a ResNet50 backbone with L2-pooling, a transformer encoder,
and a dual-path (original + flipped) forward pass. For ONNX we export
only the single-path forward (no flip), skipping normalization and
cropping which are handled in Java.

Usage:
    python scripts/tres_torch2onnx.py \
        --output_dir /mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

from pyiqa.archs.tres_arch import TReS


class TReSWrapper(nn.Module):
    """ONNX-safe wrapper for TReS.

    The original forward() includes:
    1. ImageNet normalization
    2. uniform_crop (50 random crops)
    3. Dual-path: normal + horizontally-flipped
    4. Consistency loss

    This wrapper skips all of that. It takes a single 224x224
    ImageNet-normalised crop and runs only the forward path.
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
        m = self.model

        # ResNet50 backbone
        x_bb = m.model.conv1(x)
        x_bb = m.model.bn1(x_bb)
        x_bb = m.model.relu(x_bb)
        x_bb = m.model.maxpool(x_bb)

        l1 = m.model.layer1(x_bb)   # (B, 256, 56, 56)
        l2 = m.model.layer2(l1)     # (B, 512, 28, 28)
        l3 = m.model.layer3(l2)     # (B, 1024, 14, 14)
        l4 = m.model.layer4(l3)     # (B, 2048, 7, 7)

        # L2 normalize + L2 pooling on each layer
        l1_pool = m.avg8(m.L2pooling_l1(F.normalize(l1, dim=1, p=2)))   # (B, 256, 7, 7)
        l2_pool = m.avg4(m.L2pooling_l2(F.normalize(l2, dim=1, p=2)))   # (B, 512, 7, 7)
        l3_pool = m.avg2(m.L2pooling_l3(F.normalize(l3, dim=1, p=2)))   # (B, 1024, 7, 7)
        l4_pool = m.L2pooling_l4(F.normalize(l4, dim=1, p=2))           # (B, 2048, 7, 7)

        # Concatenate features: (B, 3840, 7, 7)
        layers = torch.cat([l1_pool, l2_pool, l3_pool, l4_pool], dim=1)

        # Positional encoding
        pos_enc = m.position_embedding(torch.ones(1, layers.shape[1], 7, 7,
                                                  device=x.device))

        # Transformer encoder
        trans_out = m.transformer(layers, pos_enc)  # (B, 3840, 7, 7)

        # Average pool to 1x1
        trans_avg = m.avg7(trans_out)  # (B, 3840, 1, 1)
        trans_avg = trans_avg.flatten(1)  # (B, 3840)

        # FC projection
        fc2_out = m.fc2(trans_avg)  # (B, 2048)

        # Layer4 average pool
        l4_avg = m.avg7(l4).flatten(1)  # (B, 2048)

        # Concatenate and predict
        combined = torch.cat([fc2_out, l4_avg], dim=1)  # (B, 4096)
        score = m.fc(combined)  # (B, 1)

        return score


def main():
    parser = argparse.ArgumentParser(description='Convert TReS to ONNX')
    parser.add_argument('--pretrained', type=str, default='koniq',
                        choices=['koniq', 'flive', 'live', 'livec'],
                        help='Pretrained variant')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('TReS PyTorch → ONNX Conversion')
    print('=' * 60)

    print(f'Loading TReS (pretrained={args.pretrained})...')
    model = TReS(pretrained=args.pretrained)
    model.eval()

    wrapper = TReSWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)

    # Verify PyTorch forward works
    with torch.no_grad():
        pt_out = wrapper(dummy)
    print(f'PyTorch output: {pt_out.flatten()[0]:.4f}')

    output_path = os.path.join(args.output_dir, 'tres_model.onnx')
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

    # Verify: compare PyTorch vs ONNX on the SAME input
    print('Verifying with ONNX Runtime...')
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    test_input = dummy.numpy()
    onnx_out = sess.run(None, {'input': test_input})
    onnx_score = onnx_out[0].flatten()[0]
    pt_score = pt_out.flatten()[0].item()
    diff = abs(pt_score - onnx_score)
    print(f'PyTorch score:  {pt_score:.6f}')
    print(f'ONNX score:     {onnx_score:.6f}')
    print(f'Abs difference: {diff:.6f}')
    if diff < 0.01:
        print('PASS: PyTorch and ONNX outputs match.')
    else:
        print('FAIL: PyTorch and ONNX outputs differ too much.')
    print('Done.')


if __name__ == '__main__':
    main()
