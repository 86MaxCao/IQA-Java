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
        """Run ViT forward and return raw features from blocks 6-9.

        Returns features WITHOUT the final LayerNorm, matching what the
        forward hooks capture in the original model.
        """
        vit = self.model.vit

        # Replicate timm VisionTransformer.forward_features up to blocks
        x = vit.patch_embed(x)
        cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = vit.pos_drop(x + vit.pos_embed)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        saved = []
        for idx, blk in enumerate(vit.blocks):
            x = blk(x)
            if idx in (6, 7, 8, 9):
                saved.append(x)  # raw block output, no norm

        return saved

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) -- Inception-normalised.
        Returns:
            score: (B, 1)
        """
        feats = self._extract_vit_features(x)
        B = x.shape[0]
        H = W = 28  # 224 / 8

        # Remove CLS token, keep patch tokens (matching original extract_feature)
        feats_no_cls = [f[:, 1:, :] for f in feats]

        # Concatenate ALL 4 features (layers 6,7,8,9) on channel dim
        # -> (B, N, embed_dim*4) where N = 784, embed_dim = 768
        x = torch.cat(feats_no_cls, dim=2)

        # Stage 1: rearrange to (B, C, N) for TABlock
        x = x.transpose(1, 2)  # (B, embed_dim*4, N)
        # Reshape to spatial (B, C, H*W) -- already in this shape
        for tab in self.model.tablock1:
            x = tab(x)
        # Reshape to (B, C, H, W) for conv1
        x = x.reshape(B, -1, H, W)
        x = self.model.conv1(x)  # (B, embed_dim, H, W)
        x = self.model.swintransformer1(x)

        # Stage 2: rearrange for TABlock
        x = x.flatten(2)  # (B, C, H*W)
        for tab in self.model.tablock2:
            x = tab(x)
        x = x.reshape(B, -1, H, W)
        x = self.model.conv2(x)  # (B, embed_dim//2, H, W)
        x = self.model.swintransformer2(x)

        # Rearrange to (B, N, C) for score prediction
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim//2)

        # Score prediction (weighted sum, matching original)
        per_patch_score = self.model.fc_score(x)    # (B, N, 1)
        per_patch_weight = self.model.fc_weight(x)  # (B, N, 1)

        per_patch_score = per_patch_score.reshape(B, -1)
        per_patch_weight = per_patch_weight.reshape(B, -1)

        score = (per_patch_weight * per_patch_score).sum(dim=-1) / (
            per_patch_weight.sum(dim=-1) + 1e-8
        )
        return score.unsqueeze(1)


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
    print('MANIQA PyTorch -> ONNX Conversion')
    print('=' * 60)

    print(f'Loading MANIQA (pretrained={args.pretrained})...')
    model = MANIQA(train_dataset=args.pretrained)
    model.eval()

    wrapper = MANIQAWrapper(model)
    wrapper.eval()

    # Fixed test input for reproducible comparison
    torch.manual_seed(42)
    test_input = torch.randn(1, 3, 224, 224)

    # PyTorch inference
    with torch.no_grad():
        pytorch_score = wrapper(test_input)
    print(f'PyTorch score: {pytorch_score.item():.6f}')

    # ONNX export
    output_path = os.path.join(args.output_dir, 'maniqa_model.onnx')
    print(f'Exporting to {output_path}...')

    torch.onnx.export(
        wrapper,
        test_input,
        output_path,
        input_names=['input'],
        output_names=['score'],
        dynamic_axes={
            'input': {0: 'batch'},
            'score': {0: 'batch'},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter for ModuleList compat
    )
    print(f'Exported: {output_path}')
    onnx_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'ONNX file size: {onnx_size_mb:.1f} MB')

    # ONNX Runtime verification with the SAME input
    print('Verifying with ONNX Runtime...')
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)

    test_np = test_input.numpy()
    onnx_out = sess.run(None, {'input': test_np})
    onnx_score = float(onnx_out[0].flatten()[0])
    print(f'ONNX score:    {onnx_score:.6f}')

    # Comparison
    diff = abs(pytorch_score.item() - onnx_score)
    print(f'Abs diff:      {diff:.6f}')
    if diff < 0.01:
        print('PASS: PyTorch and ONNX outputs match (diff < 0.01)')
    else:
        print('FAIL: PyTorch and ONNX outputs differ significantly')


if __name__ == '__main__':
    main()
