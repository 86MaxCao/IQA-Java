"""
Verify MUSIQ ONNX model against PyTorch MUSIQWrapper.

Loads both models, feeds the same random 224x224 input (normalized to [-1,1]),
and compares outputs. Reports PASS if absolute difference < 0.01.
"""

import os
import sys
import numpy as np
import torch

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

# Also need the conversion script's wrapper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from musiq_torch2onnx import MUSIQWrapper

from pyiqa.archs.musiq_arch import MUSIQ

ONNX_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/musiq_model.onnx'


def main():
    print('=' * 60)
    print('MUSIQ ONNX vs PyTorch Verification')
    print('=' * 60)

    # 1. Create deterministic random input in [-1, 1]
    torch.manual_seed(42)
    np.random.seed(42)
    test_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # Clamp to [-1, 1] range as expected by the model
    test_input_np = np.clip(test_input_np, -1.0, 1.0)
    test_input_pt = torch.from_numpy(test_input_np)

    print(f'Input shape: {test_input_np.shape}')
    print(f'Input range: [{test_input_np.min():.4f}, {test_input_np.max():.4f}]')

    # 2. Load PyTorch model with MUSIQWrapper
    print('\nLoading PyTorch MUSIQ model...')
    model = MUSIQ(pretrained='koniq10k')
    model.eval()
    wrapper = MUSIQWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        pt_score = wrapper(test_input_pt).item()
    print(f'PyTorch score: {pt_score:.6f}')

    # 3. Load ONNX model
    print(f'\nLoading ONNX model from {ONNX_PATH}...')
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH)
    onnx_out = sess.run(None, {'input': test_input_np})
    onnx_score = onnx_out[0].flatten()[0]
    print(f'ONNX score:    {onnx_score:.6f}')

    # 4. Compare
    diff = abs(pt_score - onnx_score)
    print(f'\nAbsolute difference: {diff:.6f}')

    threshold = 0.01
    if diff < threshold:
        print(f'\nRESULT: PASS (diff {diff:.6f} < {threshold})')
    else:
        print(f'\nRESULT: FAIL (diff {diff:.6f} >= {threshold})')

    # Print summary
    import os
    file_size = os.path.getsize(ONNX_PATH)
    print(f'\n--- Summary ---')
    print(f'ONNX file: {ONNX_PATH}')
    print(f'ONNX size: {file_size / (1024*1024):.2f} MB')
    print(f'PyTorch score: {pt_score:.6f}')
    print(f'ONNX score:    {onnx_score:.6f}')
    print(f'Difference:    {diff:.6f}')


if __name__ == '__main__':
    main()
