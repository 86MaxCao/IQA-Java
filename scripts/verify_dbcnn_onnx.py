"""
Verify DBCNN ONNX model against PyTorch reference.

Loads the PyTorch DBCNN model via pyiqa and the exported ONNX model,
runs both on the same random input tensor, and compares scores.

Reports PASS if |pytorch_score - onnx_score| < 0.01, FAIL otherwise.
"""

import sys
import numpy as np
import torch

# Add IQA-PyTorch to path
IQA_ROOT = '/mnt/nas-tbt/caoziqi/code/experiment/IQA-PyTorch'
sys.path.insert(0, IQA_ROOT)

import pyiqa
import onnxruntime as ort

ONNX_PATH = '/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints/dbcnn_model.onnx'
THRESHOLD = 0.01


def main():
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create random test tensor (1, 3, 512, 384) in [0, 1] as pyiqa expects
    test_tensor = torch.rand(1, 3, 512, 384)

    # ---- PyTorch inference ----
    print('Loading PyTorch DBCNN via pyiqa...')
    metric = pyiqa.create_metric('dbcnn', device='cuda:0')
    metric.eval()

    with torch.no_grad():
        pytorch_score = metric(test_tensor.to('cuda:0'))
    pytorch_score_val = pytorch_score.item()
    print(f'PyTorch score: {pytorch_score_val:.6f}')

    # ---- ONNX inference ----
    # The ONNX wrapper expects already-normalised input (ImageNet mean/std),
    # matching what pyiqa does internally. We need to replicate pyiqa's
    # preprocessing to feed the same normalised tensor to the ONNX model.
    #
    # pyiqa's DBCNN preprocesses: clamp [0,1], then ImageNet normalise.
    # But our random tensor is not in [0,1]. pyiqa's forward does:
    #   x = self.default_mean / self.default_std  adjustments
    # Let's extract the normalised tensor that pyiqa feeds into the DBCNN model.

    # Actually, looking at the DBCNNWrapper - it skips built-in preprocessing.
    # The pyiqa metric does its own preprocessing before calling model.forward().
    # The ONNX model (DBCNNWrapper) expects already-normalised input.
    #
    # pyiqa's DBCNN net calls:
    #   x = (x - mean) / std  (ImageNet normalisation)
    # then feeds into features1/features2.
    #
    # But DBCNNWrapper skips this - it goes directly to features1/features2.
    # So we need to manually normalise the input for ONNX.
    #
    # However, pyiqa.create_metric wraps the model and does preprocessing
    # in its own pipeline. Let's check what preprocessing pyiqa applies.

    # The simplest approach: extract the preprocessed tensor from pyiqa
    # by hooking into the model, then feed that same tensor to ONNX.

    # Let's look at what pyiqa does. The InferenceModel wraps DBCNN and
    # applies default_mean/default_std normalization before calling net.forward().
    # DBCNN.forward() then does its own normalization again.
    #
    # Actually, let me re-examine. The DBCNNWrapper takes the place of
    # DBCNN's forward but skips DBCNN's internal normalization.
    #
    # For a fair comparison, we need to feed the ONNX model the same
    # preprocessed input that DBCNN.forward() would see after its
    # internal normalization step.
    #
    # DBCNN.forward() does:
    #   x = (x - self.default_mean.to(x)) / self.default_std.to(x)
    #   then features1(x), features2(x), bilinear pooling, fc
    #
    # So the ONNX wrapper expects input AFTER ImageNet normalization.
    # pyiqa's InferenceModel also applies normalization before calling net.
    # That means pyiqa normalizes once, then DBCNN normalizes again internally.
    #
    # For ONNX: we need to apply the SAME double normalization manually.
    # OR better: just apply what DBCNN.forward() would apply.

    # Let's just replicate the full pyiqa pipeline for the ONNX path:
    # 1. pyiqa InferenceModel preprocessing (clamp to [0,1]? + normalize)
    # 2. DBCNN internal normalization
    # Both need to be applied before feeding to ONNX wrapper.

    # Actually, the simplest correct approach: use pyiqa's model internals
    # to get the preprocessed tensor, then feed to ONNX.

    # Let me check DBCNN source to understand the exact preprocessing chain.

    # From DBCNN_arch.py forward():
    #   self.default_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    #   self.default_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    #   x = (x - self.default_mean.to(x)) / self.default_std.to(x)

    # From pyiqa InferenceModel, it wraps the metric but the DBCNN net itself
    # handles normalization in its forward(). So pyiqa passes input directly
    # to DBCNN.forward() which normalizes internally.

    # The DBCNNWrapper SKIPS this normalization - so for ONNX we need to
    # manually apply it.

    # For PyTorch reference: pyiqa passes test_tensor -> DBCNN.forward()
    #   which normalizes then computes score.
    # For ONNX: we need test_tensor -> normalize -> ONNX wrapper

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized_input = (test_tensor - mean) / std

    print(f'Loading ONNX model from {ONNX_PATH}...')
    sess = ort.InferenceSession(ONNX_PATH)
    onnx_input = normalized_input.numpy().astype(np.float32)
    onnx_out = sess.run(None, {'input': onnx_input})
    onnx_score_val = float(onnx_out[0].flatten()[0])
    print(f'ONNX score:    {onnx_score_val:.6f}')

    # ---- Compare ----
    diff = abs(pytorch_score_val - onnx_score_val)
    print(f'\nAbsolute difference: {diff:.6f}')

    import os
    onnx_size = os.path.getsize(ONNX_PATH)
    onnx_data_path = ONNX_PATH + '.data'
    onnx_data_size = os.path.getsize(onnx_data_path) if os.path.exists(onnx_data_path) else 0
    total_size = onnx_size + onnx_data_size

    if diff < THRESHOLD:
        print(f'Result: PASS')
    else:
        print(f'Result: FAIL')

    print(f'\nPyTorch score: {pytorch_score_val:.6f}')
    print(f'ONNX score:    {onnx_score_val:.6f}')
    print(f'ONNX file:     {ONNX_PATH}')
    if onnx_data_size > 0:
        print(f'ONNX data:     {onnx_data_path}')
        print(f'ONNX size:     {onnx_size / 1024:.1f} KB + {onnx_data_size / 1024 / 1024:.1f} MB data = {total_size / 1024 / 1024:.1f} MB total')
    else:
        print(f'ONNX size:     {onnx_size / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
