"""
Verify TReS ONNX model against PyTorch TReSWrapper.

Creates a deterministic test input (ImageNet-normalized random image),
runs both PyTorch wrapper and ONNX Runtime, then compares outputs.
"""

import os
import sys
import numpy as np
import torch

IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

# Also need the conversion script's TReSWrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tres_torch2onnx import TReSWrapper
from pyiqa.archs.tres_arch import TReS


def main():
    onnx_path = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/tres_model.onnx'

    print('=' * 60)
    print('TReS ONNX vs PyTorch Verification')
    print('=' * 60)

    # 1. Load PyTorch model
    print('Loading PyTorch TReS model...')
    model = TReS(pretrained='koniq')
    model.eval()
    wrapper = TReSWrapper(model)
    wrapper.eval()

    # 2. Load ONNX model
    print('Loading ONNX model...')
    import onnxruntime as ort
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider'],
    )

    # 3. Create deterministic test input (ImageNet-normalized)
    print('Creating test input (1, 3, 224, 224) with fixed seed...')
    rng = np.random.RandomState(42)
    # Simulate an ImageNet-normalized image
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    # Random pixel values in [0, 1], then normalize
    raw_image = rng.rand(1, 3, 224, 224).astype(np.float32)
    test_input = ((raw_image - imagenet_mean) / imagenet_std).astype(np.float32)

    # 4. Run PyTorch
    print('Running PyTorch inference...')
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input)
        pt_output = wrapper(pt_input)
        pt_score = pt_output.cpu().numpy().flatten()[0]

    # 5. Run ONNX
    print('Running ONNX inference...')
    onnx_output = sess.run(None, {'input': test_input})
    onnx_score = onnx_output[0].flatten()[0]

    # 6. Compare
    abs_diff = abs(float(pt_score) - float(onnx_score))

    print()
    print('-' * 40)
    print(f'PyTorch score:  {pt_score:.6f}')
    print(f'ONNX score:     {onnx_score:.6f}')
    print(f'Abs difference: {abs_diff:.6f}')
    print('-' * 40)

    import os
    onnx_size = os.path.getsize(onnx_path)
    data_path = onnx_path + '.data'
    data_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    total_size = onnx_size + data_size

    print(f'ONNX file:      {onnx_path}')
    print(f'ONNX size:      {onnx_size:,} bytes ({onnx_size/1024/1024:.1f} MB)')
    if data_size:
        print(f'ONNX data:      {data_path}')
        print(f'ONNX data size: {data_size:,} bytes ({data_size/1024/1024:.1f} MB)')
        print(f'Total size:     {total_size:,} bytes ({total_size/1024/1024:.1f} MB)')

    threshold = 0.01
    if abs_diff < threshold:
        print(f'\nPASS: Difference {abs_diff:.6f} < {threshold}')
    else:
        print(f'\nFAIL: Difference {abs_diff:.6f} >= {threshold}')

    return abs_diff < threshold


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
