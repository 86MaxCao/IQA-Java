"""
Verify CLIPIQA+ ONNX model against PyTorch reference.

Loads both models, runs them on the same random input,
and checks that the outputs match within tolerance.
"""

import os
import sys
import numpy as np
import torch

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

from pyiqa.archs.clipiqa_arch import CLIPIQA

# Also import the wrapper from the conversion script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clipiqa_torch2onnx import CLIPIQAWrapper

ONNX_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/clipiqa_model.onnx'
TOLERANCE = 0.01


def main():
    print('=' * 60)
    print('CLIPIQA+ ONNX vs PyTorch Verification')
    print('=' * 60)

    # --- Step 1: Load PyTorch model ---
    print('[1/4] Loading PyTorch CLIPIQA+ model...')
    model = CLIPIQA(model_type='clipiqa+', backbone='RN50', pretrained=True)
    model.eval()

    wrapper = CLIPIQAWrapper(model)
    wrapper.eval()
    print('       PyTorch model loaded.')

    # --- Step 2: Load ONNX model ---
    print('[2/4] Loading ONNX model...')
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    print(f'       ONNX model loaded from {ONNX_PATH}')

    # --- Step 3: Create test input ---
    # The wrapper expects CLIP-normalised input (B,3,224,224).
    # We generate a random tensor normalised with CLIP mean/std
    # to simulate a realistic input.
    print('[3/4] Creating test input (1,3,224,224)...')
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate raw image-like tensor in [0,1], then normalize with CLIP stats
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    raw_img = torch.rand(1, 3, 224, 224)
    # Normalize: (x - mean) / std
    mean_t = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    test_input = (raw_img - mean_t) / std_t
    test_input_np = test_input.numpy().astype(np.float32)
    print(f'       Input shape: {test_input.shape}, dtype: {test_input.dtype}')

    # --- Step 4: Run both models and compare ---
    print('[4/4] Running inference...')

    # PyTorch inference
    with torch.no_grad():
        pt_score = wrapper(test_input).item()

    # ONNX inference
    onnx_out = sess.run(None, {'input': test_input_np})
    onnx_score = onnx_out[0].flatten()[0]

    abs_diff = abs(pt_score - onnx_score)

    print()
    print('-' * 40)
    print(f'PyTorch score : {pt_score:.6f}')
    print(f'ONNX score    : {onnx_score:.6f}')
    print(f'Abs difference: {abs_diff:.6f}')
    print(f'Tolerance     : {TOLERANCE}')
    print('-' * 40)

    if abs_diff < TOLERANCE:
        print('Result: PASS')
    else:
        print('Result: FAIL')
        sys.exit(1)


if __name__ == '__main__':
    main()
