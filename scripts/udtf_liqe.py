"""LIQE Model UDTF Implementation

github repo link: https://github.com/zwx8981/LIQE

Cite as:
@inproceedings{zhang2023liqe,
  title={Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},
  author={Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14071--14081},
  year={2023}
}

UDTF version that extracts functions from clip and pyiqa libraries
"""

import sys
import os
import io
from datetime import datetime
from itertools import product

# UDTF imports - only available in UDTF environment
try:
    from odps.udf import annotate
    from odps.udf import BaseUDTF
    from odps.distcache import get_cache_archive, get_cache_file
    UDTF_AVAILABLE = True
except ImportError:
    # Not in UDTF environment - define dummy classes for local testing
    UDTF_AVAILABLE = False
    def annotate(*args, **kwargs):
        def decorator(cls):
            return cls
        return decorator
    
    class BaseUDTF:
        def forward(self, *args, **kwargs):
            pass
    
    def get_cache_archive(*args, **kwargs):
        raise NotImplementedError("Not in UDTF environment")
    
    def get_cache_file(*args, **kwargs):
        raise NotImplementedError("Not in UDTF environment")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import requests
import numpy as np
from PIL import Image
import torchvision.transforms.functional as torch_fun


# Constants extracted from pyiqa.archs.constants
# OPENAI_CLIP_MEAN and OPENAI_CLIP_STD values
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Quality labels and scene/distortion mappings
qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']

scenes = [
    'animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant',
    'still_life', 'others'
]

dists_map = [
    'jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast',
    'overexposure', 'underexposure', 'spatial', 'quantization', 'other'
]


def include_package_path(res_name):
    """Include package path from cached archive"""
    try:
        archive_files = get_cache_archive(res_name)
        one_file = os.path.normpath(next(archive_files).name)
        pack_dir = one_file.split(res_name)[0] + res_name + '/files/liqe/'
        print(f"Including package path: {pack_dir}")
        sys.path.insert(0, pack_dir)
        return pack_dir
    except Exception as e:
        print(f"Warning: Could not include package path for {res_name}: {e}")
        return None


# Try to include clip and pyiqa from resource files if available
try:
    include_package_path('liqe.zip')
except Exception as e:
    print(f"Warning: Could not include package path for liqe.zip: {e}")
    pass


def load_clip_model(backbone='ViT-B/32', device='cpu', model_path=None):
    """
    Load CLIP model using local clip.py implementation
    This function loads CLIP model without external dependencies on clip or pyiqa packages
    
    Parameters:
    -----------
    backbone : str
        Model name (e.g., 'ViT-B/32') or path to model file
    device : str
        Device to load model on ('cpu' or 'cuda')
    model_path : str, optional
        Direct path to model file (if provided, backbone is ignored for file loading)
    """
    try:
        # Import load function from local clip.py file in the same directory
        # This avoids dependency on installed clip or pyiqa packages
        import importlib.util
        
        # Get the directory of this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        clip_module_path = os.path.join(current_dir, 'clip.py')
        
        if not os.path.exists(clip_module_path):
            # Fallback: try to import from sys.path if clip.py is available
            raise FileNotFoundError(f"clip.py not found at {clip_module_path}")
        
        # Load the clip module dynamically
        spec = importlib.util.spec_from_file_location("clip_module", clip_module_path)
        clip_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(clip_module)
        
        # Use the load function from clip.py
        # If model_path is provided, use it directly; otherwise use backbone name
        if model_path is not None and os.path.exists(model_path):
            model = clip_module.load(model_path, device=device, jit=False)
        else:
            model = clip_module.load(backbone, device=device, jit=False)
        
        return model
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise


def load_pretrained_network(net, model_path, strict=True, param_key='params'):
    """
    Load pretrained network weights - extracted from pyiqa.archs.arch_util.load_pretrained_network
    """
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if param_key in checkpoint:
                state_dict = checkpoint[param_key]
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if exists
            state_dict_clean = {}
            for k, v in state_dict.items():
                key = k.replace('module.', '') if k.startswith('module.') else k
                state_dict_clean[key] = v
            
            net.load_state_dict(state_dict_clean, strict=strict)
            print(f"Successfully loaded pretrained model from {model_path}")
        else:
            print(f"Warning: Model path does not exist: {model_path}")
    except Exception as e:
        print(f"Error loading pretrained network: {e}")
        raise


def resize_img(img, target_width=640, target_height=360):
    """
    Resize image maintaining aspect ratio and padding to target size
    """
    try:
        original_width, original_height = img.size
        
        # Calculate aspect ratios and new dimensions
        original_aspect = original_width / original_height
        target_aspect = target_width / target_height
        
        if original_aspect > target_aspect:
            # Wider than target, resize based on width
            new_width = target_width
            new_height = round(new_width / original_aspect)
        else:
            # Higher than target, resize based on height
            new_height = target_height
            new_width = round(new_height * original_aspect)
        
        img = img.resize((new_width, new_height))
        
        # Initialize a new image with the specified background color (black)
        new_img = Image.new("RGB", (target_width, target_height))
        
        # Determine padding
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        
        # Paste the resized image onto the background
        new_img.paste(img, (pad_left, pad_top))
        return new_img
    except Exception as e:
        print(f"Error in resize_img: {e}")
        # Return a black image of target size as fallback
        return Image.new("RGB", (target_width, target_height))


class LIQE(nn.Module):
    """
    LIQE Model for image quality assessment
    Extracted and modified to work without direct clip/pyiqa dependencies
    """
    def __init__(self,
                 model_type='liqe',
                 backbone='ViT-B/32',
                 step=32,
                 num_patch=15,
                 pretrained=True,
                 pretrained_model_path=None,
                 clip_model_path=None,
                 text_feat_cache_path=None,
                 mtl=False,
                 ) -> None:
        super().__init__()
        
        # Replace assert with try-except
        try:
            if backbone != 'ViT-B/32':
                raise ValueError(f'Only support ViT-B/32 now, got {backbone}')
        except ValueError as e:
            print(f"Error: {e}")
            raise
        
        self.backbone = backbone
        
        # Load CLIP model
        try:
            self.clip_model = load_clip_model(
                backbone=self.backbone,
                device='cpu',
                model_path=clip_model_path
            )
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise ValueError(f"Error loading CLIP model: {e}")
        
        self.model_type = model_type
        
        # Set default mean and std for image normalization
        # These are extracted from pyiqa.archs.constants
        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
        
        # Disable gradient for logit_scale (temperature parameter)
        try:
            self.clip_model.logit_scale.requires_grad = False
        except AttributeError:
            print("Warning: logit_scale not found in CLIP model")
        
        self.step = step
        self.num_patch = num_patch
        
        # Load pretrained weights
        if pretrained_model_path is None and pretrained:
            # In UDTF, we should get model path from cache file
            # For local testing, skip this step
            try:
                # Try to get from UDTF cache (only works in UDTF environment)
                pretrained_model_path = get_cache_file('liqe_mix.pt').name
            except (NameError, AttributeError):
                # Not in UDTF environment, skip
                print("Warning: Not in UDTF environment, pretrained_model_path is None")
            except Exception as e:
                print(f"Warning: Could not get model from cache: {e}")
        
        if pretrained_model_path is not None:
            try:
                load_pretrained_network(self, pretrained_model_path, True, 'params')
            except Exception as e:
                print(f"Warning: Could not load pretrained network: {e}")
        
        # Determine if MTL (multi-task learning) mode
        if pretrained == 'mix':
            self.mtl = True
            if text_feat_cache_path is None:
                try:
                    # Try to get from UDTF cache (only works in UDTF environment)
                    text_feat_cache_path = get_cache_file('liqe_text_feat_mix.pt').name
                except (NameError, AttributeError):
                    # Not in UDTF environment, skip
                    print("Warning: Not in UDTF environment, text_feat_cache_path is None")
                    text_feat_cache_path = None
                except Exception as e:
                    text_feat_cache_path = None
                    print(f"Warning: Could not get text features from cache: {e}")
        else:
            self.mtl = mtl
            if text_feat_cache_path is None:
                try:
                    # Try to get from UDTF cache (only works in UDTF environment)
                    text_feat_cache_path = get_cache_file('liqe_text_feat.pt').name
                except (NameError, AttributeError):
                    # Not in UDTF environment, skip
                    print("Warning: Not in UDTF environment, text_feat_cache_path is None")
                    text_feat_cache_path = None
                except Exception as e:
                    print(f"Warning: Could not get text features from cache: {e}")
                    text_feat_cache_path = None
        
        # Load text features from cache (must be provided)
        if text_feat_cache_path and os.path.exists(text_feat_cache_path):
            try:
                self.text_features = torch.load(
                    text_feat_cache_path, map_location='cpu'
                )
                print(f"Successfully loaded text features from {text_feat_cache_path}")
            except Exception as e:
                print(f"Error loading text features from cache: {e}")
                raise ValueError(
                    "Text features must be provided via text_feat_cache_path"
                )
        else:
            raise ValueError(
                f"Cannot find text features file: {text_feat_cache_path}. "
                "Text features must be pre-computed and provided."
            )

    def forward(self, x, only_score=True):
        """Forward pass through LIQE model"""
        try:
            bs = x.size(0)
            h = x.size(2)
            w = x.size(3)
            
            # Replace assert with try-except
            if h < 224 or w < 224:
                raise ValueError(f'Short side is less than 224 (got h={h}, w={w}), try upsampling the original image')
            
            # Preprocess image
            x = (x - self.default_mean.to(x)) / self.default_std.to(x)
            
            # Extract patches
            x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, 3, 224, 224)
            
            # Select patches
            if x.size(1) < self.num_patch:
                num_patch = x.size(1)
                self.num_patch = num_patch
            else:
                num_patch = self.num_patch
            
            if self.training:
                sel = torch.randint(low=0, high=x.size(1), size=(num_patch,))
            else:
                sel_step = max(1, x.size(1) // self.num_patch)
                sel = torch.zeros(num_patch)
                for i in range(num_patch):
                    sel[i] = sel_step * i
                sel = sel.long()
            
            x = x[:, sel, ...]
            x = x.reshape(bs, num_patch, x.shape[2], x.shape[3], x.shape[4])
            
            # Get image features
            text_features = self.text_features.to(x.device)
            x = x.view(bs * x.size(1), x.size(2), x.size(3), x.size(4))
            
            try:
                image_features = self.clip_model.encode_image(x, pos_embedding=True)
            except TypeError:
                # Some CLIP models don't have pos_embedding parameter
                image_features = self.clip_model.encode_image(x)
            
            # Normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Cosine similarity as logits
            logit_scale = self.clip_model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            
            logits_per_image = logits_per_image.view(bs, self.num_patch, -1)
            logits_per_image = logits_per_image.mean(1)
            logits_per_image = F.softmax(logits_per_image, dim=1)
            
            if only_score:
                if self.mtl:
                    logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
                else:
                    logits_per_image = logits_per_image.view(-1, len(qualitys))
                logits_quality = logits_per_image.sum(3).sum(2) if self.mtl else logits_per_image
                quality = (1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 
                          3 * logits_quality[:, 2] + 4 * logits_quality[:, 3] + 
                          5 * logits_quality[:, 4])
                # Return tensor (not .item()) to support batch processing
                return quality
            
            if self.mtl:
                logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
                
                logits_quality = logits_per_image.sum(3).sum(2)
                similarity_scene = logits_per_image.sum(3).sum(1)
                similarity_distortion = logits_per_image.sum(1).sum(1)
                
                distortion_index = similarity_distortion.argmax(dim=1)
                scene_index = similarity_scene.argmax(dim=1)
                quality = (1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 
                          3 * logits_quality[:, 2] + 4 * logits_quality[:, 3] + 
                          5 * logits_quality[:, 4])
                
                pred = 'A photo of {} with {} artifacts, which has a perceptual quality of {}'.format(
                    scenes[scene_index.item()], dists_map[distortion_index.item()], quality.item())
                return quality, pred
            else:
                logits_per_image = logits_per_image.view(-1, len(qualitys))
                logits_quality = logits_per_image
                
                quality = (1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 
                          3 * logits_quality[:, 2] + 4 * logits_quality[:, 3] + 
                          5 * logits_quality[:, 4])
                
                return quality, 'A photo has a perceptual quality of {}'.format(quality.item())
        except Exception as e:
            print(f"Error in LIQE forward pass: {e}")
            raise


class ImageData(data.Dataset):
    """
    Image data loader for UDTF
    Handles loading images from URLs
    """
    def __init__(self, urls: list):
        super(ImageData, self).__init__()
        self.urls = urls
        self.dft_img = torch.zeros(3, 640, 360, dtype=torch.float32)
    
    def __len__(self):
        return len(self.urls)
    
    def __getitem__(self, idx):
        """Load image from URL and preprocess"""
        image_url = self.urls[idx]
        raw_image = None
        
        # Try to fetch image (up to 3 retries)
        for i in range(3):
            try:
                res = requests.get(image_url, timeout=10)
                if res.status_code == 200:
                    raw_image = res.content
                    break
            except Exception as e:
                print(f"Error fetching image (attempt {i+1}/3): {e}")
                continue
        
        if raw_image is None:
            print(f'image[{image_url}] is null')
            return image_url, self.dft_img
        
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(raw_image)).convert('RGB')
            image = resize_img(image)
            
            # Convert to tensor
            img_tensor = torch_fun.to_tensor(image)
            
            return image_url, img_tensor
        except Exception as e:
            print(f'Error processing image[{image_url}]: {e}')
            return image_url, self.dft_img


@annotate('array<string> -> string, double')
class UDTFLIQE(BaseUDTF):
    """
    UDTF for LIQE image quality assessment
    Takes an array of image URLs and returns quality scores
    """
    def __init__(self):
        # Important: limit torch threads for performance
        torch.set_num_threads(1)
        
        try:
            # Load model and text features from cache
            model_path = get_cache_file('liqe_mix.pt').name
            clip_model_path = get_cache_file('ViT-B-32.pt').name
            text_feat_path = get_cache_file('liqe_text_feat_mix.pt').name
            
            # Initialize LIQE model
            self.model = LIQE(
                pretrained='mix',
                pretrained_model_path=model_path,
                clip_model_path=clip_model_path,
                text_feat_cache_path=text_feat_path
            ).eval()
            
            self.counter = 0
            print("LIQE model initialized successfully")
        except Exception as e:
            print(f"Error initializing LIQE model: {e}")
            self.model = None
    
    def process(self, image_urls):
        """
        Process a batch of image URLs
        IMPORTANT: Must ensure output count equals input count (len(image_urls))
        Uses -1.0 as error indicator (invalid score) to distinguish from normal low scores
        """
        # Track which URLs have been processed
        processed_urls = set()
        # Use -1.0 as error indicator (invalid score, easy to identify)
        ERROR_SCORE = -1.0
        
        if self.model is None:
            print("Error: Model not initialized")
            # Output error score for all URLs to maintain count consistency
            for url in image_urls:
                try:
                    self.forward(url, ERROR_SCORE)
                    processed_urls.add(url)
                except Exception as e:
                    print(f"Critical error: Cannot forward result for {url}: {e}")
            return
        
        try:
            print(f'counter: {self.counter}')
            print(f'urls length: {len(image_urls)}')
            sys.stdout.flush()
            
            self.counter += 1
            
            # Create image data loader
            image_data = ImageData(image_urls)
            data_loader = torch.utils.data.DataLoader(
                image_data, 
                batch_size=10, 
                shuffle=False, 
                num_workers=0
            )
            
            # Process batches
            for idx, batch in enumerate(data_loader):
                batch_urls = []
                batch_scores = []
                
                try:
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: processing batch {idx}")
                    sys.stdout.flush()
                    
                    urls, images = batch[0], batch[1]
                    batch_urls = list(urls)
                    
                    # Check if any images are default (failed to load)
                    # Default image is all zeros (black image), check if mean is 0
                    valid_images = []
                    for img_idx, img in enumerate(images):
                        # Check if image is default (all zeros) - indicates load failure
                        if torch.allclose(img, torch.zeros_like(img), atol=1e-6):
                            valid_images.append(False)
                            print(f"Warning: Image for URL {batch_urls[img_idx]} is default (load failed)")
                        else:
                            valid_images.append(True)
                    
                    # Process images through model
                    with torch.no_grad():
                        try:
                            # Model returns quality_scores tensor when only_score=True
                            # For UDTF, we only need scores, not predictions
                            quality_scores = self.model(images, only_score=True)
                            
                            # Ensure quality_scores is a tensor
                            if not isinstance(quality_scores, torch.Tensor):
                                # Convert to tensor if needed
                                quality_scores = torch.tensor([quality_scores])
                            elif quality_scores.dim() == 0:
                                # Scalar tensor, expand to 1D
                                quality_scores = quality_scores.unsqueeze(0)
                            
                            # Extract scores for each URL in batch
                            for k in range(len(batch_urls)):
                                # If image failed to load, mark as error
                                if not valid_images[k]:
                                    batch_scores.append(ERROR_SCORE)
                                    continue
                                
                                try:
                                    if isinstance(quality_scores, torch.Tensor):
                                        if quality_scores.dim() > 0 and k < len(quality_scores):
                                            score = float(quality_scores[k].item())
                                            # Validate score is in valid range (1-5)
                                            if score < 1.0 or score > 5.0:
                                                print(f"Warning: Score {score} out of range for URL {batch_urls[k]}")
                                                score = ERROR_SCORE
                                        elif quality_scores.dim() == 0:
                                            # Scalar tensor, use for first item only
                                            score = float(quality_scores.item()) if k == 0 else ERROR_SCORE
                                            if score < 1.0 or score > 5.0:
                                                score = ERROR_SCORE
                                        else:
                                            # Index out of range
                                            score = ERROR_SCORE
                                    else:
                                        # Not a tensor
                                        score = float(quality_scores) if k == 0 else ERROR_SCORE
                                        if score < 1.0 or score > 5.0:
                                            score = ERROR_SCORE
                                    batch_scores.append(score)
                                except Exception as e:
                                    print(f"Error extracting score for URL {batch_urls[k]}: {e}")
                                    batch_scores.append(ERROR_SCORE)
                                    
                        except Exception as e:
                            print(f"Error in model inference for batch {idx}: {e}")
                            # Mark all URLs in this batch as failed
                            batch_scores = [ERROR_SCORE] * len(batch_urls)
                    
                    # Forward results for this batch
                    for url, score in zip(batch_urls, batch_scores):
                        try:
                            self.forward(url, score)
                            processed_urls.add(url)
                        except Exception as e:
                            print(f"Error forwarding result for {url}: {e}")
                            # Try again with error score
                            try:
                                self.forward(url, ERROR_SCORE)
                                processed_urls.add(url)
                            except Exception as e2:
                                print(f"Critical error: Cannot forward result for {url}: {e2}")
                
                except Exception as e:
                    print(f"Error processing batch {idx}: {e}")
                    # Forward error scores for all URLs in this batch
                    batch_start_idx = idx * 10
                    batch_end_idx = min((idx + 1) * 10, len(image_urls))
                    for url in image_urls[batch_start_idx:batch_end_idx]:
                        if url not in processed_urls:
                            try:
                                self.forward(url, ERROR_SCORE)
                                processed_urls.add(url)
                            except Exception as e2:
                                print(f"Critical error: Cannot forward result for {url}: {e2}")
        
        except Exception as e:
            print(f"Error in process method: {e}")
            # Forward error scores for all unprocessed URLs
            for url in image_urls:
                if url not in processed_urls:
                    try:
                        self.forward(url, ERROR_SCORE)
                        processed_urls.add(url)
                    except Exception as e2:
                        print(f"Critical error: Cannot forward result for {url}: {e2}")
        
        # Final check: ensure all URLs have been processed
        unprocessed = set(image_urls) - processed_urls
        if unprocessed:
            print(f"Warning: {len(unprocessed)} URLs were not processed, outputting error scores")
            for url in unprocessed:
                try:
                    self.forward(url, ERROR_SCORE)
                except Exception as e:
                    print(f"Critical error: Cannot forward result for {url}: {e}")


if __name__ == '__main__':
    # Local testing mode - no UDTF dependencies
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LIQE model locally')
    parser.add_argument(
        '--model_path',
        type=str,
        default='weights/liqe_mix.pt',
        help='Path to LIQE model weights'
    )
    parser.add_argument(
        '--clip_model_path',
        type=str,
        default='weights/ViT-B-32.pt',
        help='Path to CLIP model weights'
    )
    parser.add_argument(
        '--text_feat_path',
        type=str,
        default='weights/liqe_text_feat_mix.pt',
        help='Path to text features cache file'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        default='http://dzlk-cloud.oss-cn-zhangjiakou-gd.aliyuncs.com/zlkoss/socol/20250507/fc673e41c5384cdeb23b3cfd21ff2657/52196787/2025050714/1746598676496_143014e2-2b80-4124-b1c3-e57da2c43373.jpg',
        help='Path or URL to test image'
    )
    parser.add_argument(
        '--batch_images',
        nargs='+',
        help='Multiple image paths to test in batch'
    )
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Resolve model paths
    if not os.path.isabs(args.model_path):
        model_path = os.path.join(project_root, args.model_path)
    else:
        model_path = args.model_path
    
    if not os.path.isabs(args.clip_model_path):
        clip_model_path = os.path.join(project_root, args.clip_model_path)
    else:
        clip_model_path = args.clip_model_path
    
    if not os.path.isabs(args.text_feat_path):
        text_feat_path = os.path.join(project_root, args.text_feat_path)
    else:
        text_feat_path = args.text_feat_path
    
    print(f"Loading model from: {model_path}")
    print(f"Loading text features from: {text_feat_path}")
    
    # Initialize model
    try:
        model = LIQE(
            pretrained='mix',
            pretrained_model_path=model_path,
            clip_model_path=clip_model_path,
            text_feat_cache_path=text_feat_path
        ).eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Process images
    if args.batch_images:
        image_paths = args.batch_images
    else:
        image_paths = [args.image_path]
    
    for img_path in image_paths:
        # Check if it's a URL or local file path
        if img_path.startswith('http://') or img_path.startswith('https://'):
            # Download image from URL
            try:
                import requests
                print(f"\nDownloading image from URL: {img_path}")
                response = requests.get(img_path, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                print(f"Image downloaded successfully")
            except Exception as e:
                print(f"Error downloading image from URL {img_path}: {e}")
                continue
        else:
            # Resolve local image path
            if not os.path.isabs(img_path):
                img_path = os.path.join(project_root, img_path)
            
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            
            try:
                # Load and preprocess image from local file
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        try:
            # Preprocess image
            img = resize_img(img)
            
            # Convert to tensor
            img_tensor = torch_fun.to_tensor(img).unsqueeze(0)
            
            print(f"\nProcessing image: {img_path}")
            print(f"Image shape: {img_tensor.shape}")
            
            # Run inference
            with torch.no_grad():
                quality, prediction = model(img_tensor)
            
            print(f"Quality score: {quality.item():.4f}")
            print(f"Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            import traceback
            traceback.print_exc()

