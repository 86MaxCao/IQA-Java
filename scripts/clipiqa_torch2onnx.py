"""
Convert CLIPIQA+ PyTorch model to ONNX format.

CLIPIQA+ uses a CLIP image encoder (RN50) with learned text prompts.
For ONNX we precompute the text features from the PromptLearner and
bake them into the model as a constant buffer, exporting only the
image encoder + cosine similarity scoring path.

Usage:
    python scripts/clipiqa_torch2onnx.py \
        --output_dir /mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

from pyiqa.archs.clipiqa_arch import CLIPIQA


class CLIPIQAWrapper(nn.Module):
    """ONNX-safe wrapper for CLIPIQA+.

    Precomputes text features from the learned PromptLearner and stores
    them as a buffer. At inference time, only the CLIP image encoder
    runs, followed by cosine similarity with the frozen text features.

    CLIP normalization is expected to be done externally (Java side).
    """

    def __init__(self, model):
        super().__init__()
        self.clip_model = model.clip_model[0]
        self.logit_scale = self.clip_model.logit_scale

        # Precompute text features
        with torch.no_grad():
            text_features = model.prompt_learner(self.clip_model)
        self.register_buffer('text_features', text_features)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) — already CLIP-normalised.
        Returns:
            score: (B, 1)
        """
        # Encode image
        image_features = self.clip_model.encode_image(x, pos_embedding=False)

        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity with temperature
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        # logits: (B, num_prompts) where num_prompts=2 for clipiqa+

        # Reshape to (B, num_pairs, 2), softmax over last dim
        logits = logits.view(logits.shape[0], -1, 2)  # (B, 1, 2)
        probs = F.softmax(logits, dim=-1)

        # Score = mean probability of "good quality" class (index 0)
        score = probs[:, :, 0].mean(dim=1, keepdim=True)  # (B, 1)

        return score


def main():
    parser = argparse.ArgumentParser(description='Convert CLIPIQA+ to ONNX')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/',
                        help='Output directory')
    parser.add_argument('--save_text_features', action='store_true',
                        help='Also save text features as JSON')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 60)
    print('CLIPIQA+ PyTorch → ONNX Conversion')
    print('=' * 60)

    print('Loading CLIPIQA+ (RN50, clipiqa+)...')
    model = CLIPIQA(pretrained='clipiqa+', backbone='RN50')
    model.eval()

    wrapper = CLIPIQAWrapper(model)
    wrapper.eval()

    # Save text features as JSON for reference
    if args.save_text_features:
        text_feat_path = os.path.join(args.output_dir, 'clipiqa_text_features.json')
        text_features_np = wrapper.text_features.cpu().numpy().tolist()
        with open(text_feat_path, 'w') as f:
            json.dump({'text_features': text_features_np}, f)
        print(f'Saved text features: {text_feat_path}')

    dummy = torch.randn(1, 3, 224, 224)

    # Verify PyTorch forward works
    with torch.no_grad():
        pt_out = wrapper(dummy)
    print(f'PyTorch output: {pt_out.flatten()[0]:.4f}')

    output_path = os.path.join(args.output_dir, 'clipiqa_model.onnx')
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
