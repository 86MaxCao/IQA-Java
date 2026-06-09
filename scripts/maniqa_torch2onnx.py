"""
Convert MANIQA PyTorch model to ONNX format.

MANIQA uses a ViT backbone with forward hooks to capture intermediate
features. For ONNX export we replace the hook-based extraction with
explicit layer indexing so the graph is fully traceable.

Usage:
    python scripts/maniqa_torch2onnx.py \
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

from pyiqa.archs.maniqa_arch import MANIQA


class MANIQAWrapper(nn.Module):
    """ONNX-safe wrapper for MANIQA.

    The original forward() uses ``SaveOutput`` hooks on ViT blocks 6-9.
    Hooks are invisible to the ONNX tracer, so we manually run the ViT
    encoder and collect the required intermediate outputs.

    Normalization (Inception mean/std 0.5) is expected to be done
    externally (Java side).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def _extract_vit_features(self, x):
        """Run ViT forward and return features from layers 6-9."""
        vit = self.model.vit

        # Patch embedding
        x = vit.patch_embed(x)
        cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = vit.pos_drop(x + vit.pos_embed)

        saved = []
        for idx, blk in enumerate(vit.blocks):
            x = blk(x)
            if idx in (6, 7, 8, 9):
                saved.append(vit.norm(x))

        return saved

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) — Inception-normalised.
        Returns:
            score: (B, 1)
        """
        feats = self._extract_vit_features(x)

        # Remove CLS token, keep patch tokens
        B = x.shape[0]
        feats_no_cls = [f[:, 1:, :] for f in feats]

        # Concat features from layers 6-7 and 8-9 (like original code)
        x1 = torch.cat(feats_no_cls[:2], dim=2)  # layers 6,7
        x2 = torch.cat(feats_no_cls[2:], dim=2)  # layers 8,9

        # TABlock + SwinTransformer stages
        x1 = self.model.tablock1(x1)
        x2 = self.model.tablock2(x2)

        # Reshape to spatial: (B, H, W, C) where H=W=28 for patch8/224
        H = W = 28  # 224 / 8
        x1 = x1.transpose(1, 2).reshape(B, -1, H, W)
        x2 = x2.transpose(1, 2).reshape(B, -1, H, W)

        x1 = self.model.swintransformer1(x1)
        x2 = self.model.swintransformer2(x2)

        x1 = x1.flatten(2).transpose(1, 2)  # (B, N, C)
        x2 = x2.flatten(2).transpose(1, 2)

        # Score prediction (weighted sum)
        q1 = self.model.fc_score(x1)   # (B, N, 1)
        q2 = self.model.fc_score(x2)
        w1 = self.model.fc_weight(x1)  # (B, N, 1)
        w2 = self.model.fc_weight(x2)

        q = (q1 * w1 + q2 * w2) / (w1 + w2 + 1e-8)
        score = q.mean(dim=1)  # (B, 1)

        return score


def main():
    parser = argparse.ArgumentParser(description='Convert MANIQA to ONNX')
    parser.add_argument('--pretrained', type=str, default='koniq',
                        choices=['pipal', 'koniq', 'kadid'],
                        help='Pretrained variant')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('MANIQA PyTorch → ONNX Conversion')
    print('=' * 60)

    print(f'Loading MANIQA (pretrained={args.pretrained})...')
    model = MANIQA(train_dataset=args.pretrained)
    model.eval()

    wrapper = MANIQAWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)

    output_path = os.path.join(args.output_dir, 'maniqa_model.onnx')
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
