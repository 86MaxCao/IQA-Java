"""
Verify MANIQA ONNX model against PyTorch model.

Compares outputs from the ONNX wrapper (which expects Inception-normalised
input) with the same wrapper run in PyTorch, using the same random input.
"""

import os
import sys
import numpy as np
import torch

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ONNX_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/maniqa_model.onnx'
DEVICE = 'cuda:0'  # CUDA_VISIBLE_DEVICES=1 maps physical GPU 1 to cuda:0


def main():
    print('=' * 60)
    print('MANIQA ONNX vs PyTorch Verification')
    print('=' * 60)

    # ---- 1. Load PyTorch model (same wrapper used for ONNX export) ----
    from pyiqa.archs.maniqa_arch import MANIQA
    from maniqa_torch2onnx import MANIQAWrapper

    print('Loading PyTorch MANIQA model...')
    model = MANIQA(train_dataset='koniq')
    model.eval()
    wrapper = MANIQAWrapper(model)
    wrapper.eval()
    wrapper = wrapper.to(DEVICE)

    # ---- 2. Load ONNX model ----
    print(f'Loading ONNX model from {ONNX_PATH}...')
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # ---- 3. Create test input ----
    # Inception normalisation: mean=0.5, std=0.5 for all channels
    # The wrapper expects already-normalised input (normalisation done externally)
    torch.manual_seed(42)
    np.random.seed(42)
    raw_input = torch.rand(1, 3, 224, 224)  # [0, 1] range
    # Apply Inception normalisation: (x - 0.5) / 0.5 = 2*x - 1
    normalised_input = (raw_input - 0.5) / 0.5

    print(f'Input shape: {normalised_input.shape}')
    print(f'Input range: [{normalised_input.min():.4f}, {normalised_input.max():.4f}]')

    # ---- 4. Run PyTorch model ----
    print('\nRunning PyTorch model...')
    with torch.no_grad():
        pytorch_input = normalised_input.to(DEVICE)
        pytorch_score = wrapper(pytorch_input)
    pytorch_score_val = pytorch_score.cpu().numpy().flatten()[0]
    print(f'PyTorch score: {pytorch_score_val:.6f}')

    # ---- 5. Run ONNX model ----
    print('Running ONNX model...')
    onnx_input = normalised_input.numpy().astype(np.float32)
    onnx_output = sess.run(None, {'input': onnx_input})
    onnx_score_val = onnx_output[0].flatten()[0]
    print(f'ONNX score:    {onnx_score_val:.6f}')

    # ---- 6. Compare ----
    abs_diff = abs(float(pytorch_score_val) - float(onnx_score_val))
    print(f'\nAbsolute difference: {abs_diff:.8f}')

    threshold = 0.01
    if abs_diff < threshold:
        print(f'\nRESULT: PASS (difference {abs_diff:.8f} < {threshold})')
    else:
        print(f'\nRESULT: FAIL (difference {abs_diff:.8f} >= {threshold})')

    return abs_diff < threshold


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
