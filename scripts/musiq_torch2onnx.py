"""
Convert MUSIQ PyTorch model to ONNX format.

MUSIQ uses multi-scale patches with spatial/scale position embeddings.
For ONNX we simplify to single-scale fixed-size (224x224) input, manually
constructing the patch tokens, positions, and masks that the transformer
encoder expects.

Usage:
    python scripts/musiq_torch2onnx.py \
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

from pyiqa.archs.musiq_arch import MUSIQ


class MUSIQWrapper(nn.Module):
    """ONNX-safe wrapper for MUSIQ.

    The original forward() calls ``get_multiscale_patches`` which uses
    dynamic control flow unsuitable for ONNX tracing. This wrapper
    manually extracts non-overlapping patches from a fixed 224x224 input,
    constructs the required spatial positions, scale positions, and masks,
    then runs through the CNN stem, embedding, transformer, and head.

    Normalization to [-1, 1] is expected to be done externally (Java side).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.patch_size = 32
        self.hidden_size = 384
        self.spatial_pos_grid_size = 10

    def _extract_patches(self, x):
        """Extract non-overlapping 32x32 patches from 224x224 input.

        Returns:
            patches: (B, num_patches, 3, 32, 32)
            spatial_positions: (B, num_patches) — hashed grid positions
            scale_positions: (B, num_patches) — all zeros (single scale)
            masks: (B, num_patches) — all ones
        """
        B, C, H, W = x.shape
        pH = H // self.patch_size  # 7
        pW = W // self.patch_size  # 7
        num_patches = pH * pW     # 49

        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)
        # (B, 3, pH, pW, 32, 32)
        patches = patches.contiguous().view(B, C, num_patches, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # (B, num_patches, 3, 32, 32)

        # Spatial positions: hash from grid coordinates
        # Original uses: hash(row, col) % (grid_size^2)
        grid_h = torch.arange(pH, device=x.device).float() / pH * self.spatial_pos_grid_size
        grid_w = torch.arange(pW, device=x.device).float() / pW * self.spatial_pos_grid_size
        gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')
        spatial_pos = (gh.long() * self.spatial_pos_grid_size + gw.long()) \
                      .view(num_patches)
        spatial_positions = spatial_pos.unsqueeze(0).expand(B, -1)  # (B, num_patches)

        scale_positions = torch.zeros(B, num_patches, dtype=torch.long, device=x.device)
        masks = torch.ones(B, num_patches, dtype=torch.bool, device=x.device)

        return patches, spatial_positions, scale_positions, masks

    def _patch_to_token(self, patches):
        """Run patches through CNN stem and embedding layer.

        Args:
            patches: (B, num_patches, 3, 32, 32)
        Returns:
            tokens: (B, num_patches, hidden_size)
        """
        B, N, C, pH, pW = patches.shape
        x = patches.view(B * N, C, pH, pW)

        # CNN stem: conv_root -> gn_root -> relu -> root_pool -> block1
        x = self.model.conv_root(x)
        x = self.model.gn_root(x)
        x = nn.functional.relu(x, inplace=False)
        x = self.model.root_pool(x)
        x = self.model.block1(x)

        # Flatten spatial dims for embedding
        # block1 output: (B*N, 256, h, w) where h=w=patch_size/4=8
        x = x.permute(0, 2, 3, 1)  # (B*N, h, w, 256) — TF channel order
        x = x.contiguous().view(B * N, -1)  # (B*N, 256*8*8=16384)
        x = self.model.embedding(x)  # (B*N, 384)
        x = x.view(B, N, self.hidden_size)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) — already normalized to [-1, 1].
        Returns:
            score: (B, 1)
        """
        patches, spatial_pos, scale_pos, masks = self._extract_patches(x)
        tokens = self._patch_to_token(patches)

        # Transformer encoder expects (tokens, spatial_pos, scale_pos, masks)
        # but internally prepends CLS token
        x = self.model.transformer_encoder(tokens, spatial_pos, scale_pos, masks)

        # Head on CLS token (index 0)
        score = self.model.head(x[:, 0, :])  # (B, 1)
        return score


def main():
    parser = argparse.ArgumentParser(description='Convert MUSIQ to ONNX')
    parser.add_argument('--pretrained', type=str, default='koniq10k',
                        choices=['koniq10k', 'ava', 'paq2piq', 'spaq',
                                 'imagenet_pretrain'],
                        help='Pretrained variant')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('MUSIQ PyTorch → ONNX Conversion')
    print('=' * 60)

    print(f'Loading MUSIQ (pretrained={args.pretrained})...')
    model = MUSIQ(pretrained=args.pretrained)
    model.eval()

    wrapper = MUSIQWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 224, 224)

    # Verify PyTorch forward works
    with torch.no_grad():
        pt_out = wrapper(dummy)
    print(f'PyTorch output: {pt_out.flatten()[0]:.4f}')

    output_path = os.path.join(args.output_dir, 'musiq_model.onnx')
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
