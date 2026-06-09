"""
Upload IQA ONNX models to Hugging Face Hub.

Usage:
    export HF_TOKEN=your_token
    python scripts/upload_onnx_to_hf.py --repo_id 86Cao/IQA-ONNX-Models
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo


CHECKPOINT_DIR = "/mnt/nas-tbt/tbt/checkpoint/torch/hub/checkpoints"

MODEL_FILES = [
    # LIQE
    "clip_model.onnx",
    "clip_model.onnx.data",
    "liqe_model.onnx",
    "liqe_model.onnx.data",
    "text_features.json",
    # DBCNN
    "dbcnn_model.onnx",
    "dbcnn_model.onnx.data",
    # HyperIQA
    "hyperiqa_model.onnx",
    "hyperiqa_model.onnx.data",
    # MANIQA
    "maniqa_model.onnx",
    # MUSIQ
    "musiq_model.onnx",
    "musiq_model.onnx.data",
    # TReS
    "tres_model.onnx",
    "tres_model.onnx.data",
    # CLIPIQA
    "clipiqa_model.onnx",
    "clipiqa_model.onnx.data",
]


def main():
    parser = argparse.ArgumentParser(description="Upload IQA ONNX models to HuggingFace")
    parser.add_argument("--repo_id", type=str, default="86Cao/IQA-ONNX-Models")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        return

    api = HfApi(token=token)

    print(f"Creating/checking repository: {args.repo_id}")
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    print(f"Repository ready: {args.repo_id}")

    missing = []
    for f in MODEL_FILES:
        path = os.path.join(args.checkpoint_dir, f)
        if not os.path.exists(path):
            missing.append(f)

    if missing:
        print(f"\nWARNING: {len(missing)} files missing:")
        for f in missing:
            print(f"  - {f}")
        resp = input("Continue uploading available files? [y/N] ")
        if resp.lower() != "y":
            return

    uploaded = 0
    failed = 0
    for f in MODEL_FILES:
        path = os.path.join(args.checkpoint_dir, f)
        if not os.path.exists(path):
            continue
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"Uploading {f} ({size_mb:.1f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f,
                repo_id=args.repo_id,
                repo_type="model",
                token=token,
            )
            uploaded += 1
            print(f"  OK: {f}")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {f} - {e}")

    print(f"\nDone: {uploaded} uploaded, {failed} failed")
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
