"""LIQE Model

github repo link: https://github.com/zwx8981/LIQE

Cite as:
@inproceedings{zhang2023liqe,
  title={Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective},
  author={Zhang, Weixia and Zhai, Guangtao and Wei, Ying and Yang, Xiaokang and Ma, Kede},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14071--14081},
  year={2023}
}

"""

import os
from itertools import product

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from pyiqa.archs.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.archs.clip_model import load
from pyiqa.archs.arch_util import load_pretrained_network


qualitys = ['bad', 'poor', 'fair', 'good', 'perfect']
scenes = ['animal', 'cityscape', 'human', 'indoor', 'landscape', 'night', 'plant', 'still_life', 'others']
dists_map = ['jpeg2000 compression', 'jpeg compression', 'noise', 'blur', 'color', 'contrast', 'overexposure',
            'underexposure', 'spatial', 'quantization', 'other']

default_model_urls = {'koniq': 'https://github.com/zwx8981/IQA-PyTorch/releases/download/Weights/liqe_koniq.pt',
                      'mix': 'https://github.com/zwx8981/IQA-PyTorch/releases/download/Weights/liqe_mix.pt'}


def resize_img(img, target_width=640, target_height=360):
    original_width, original_height = img.size

    # Calculate aspect ratios and new dimensions
    original_aspect = original_width / original_height
    target_aspect = target_width /target_height

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


class LIQE(nn.Module):
    def __init__(self,
                 model_type='liqe',
                 backbone = 'ViT-B/32',
                 step = 32,
                 num_patch = 15,
                 pretrained=True,
                 pretrained_model_path=None,
                 text_feat_cache_path=None,
                #  device="cpu",
                 mtl = False,
                 ) -> None:
        super().__init__()
        assert backbone == 'ViT-B/32', 'Only support ViT-B/32 now'
        # self.device = device
        self.backbone = backbone
        self.clip_model = load(self.backbone, 'cpu')  # avoid saving clip weights
        self.model_type = model_type

        self.default_mean = torch.Tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
        self.default_std = torch.Tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)

        self.clip_model.logit_scale.requires_grad = False

        self.step = step
        self.num_patch = num_patch

        if pretrained_model_path is None and pretrained:
            url_key = 'koniq' if isinstance(pretrained, bool) else pretrained
            pretrained_model_path = default_model_urls[url_key]
        if pretrained_model_path is not None:
            load_pretrained_network(self, pretrained_model_path, True, 'params')

        if pretrained == 'mix':
            self.mtl = True
            if text_feat_cache_path is None:
                text_feat_cache_path = os.path.expanduser("~/.cache/pyiqa/liqe_text_feat_mix.pt")
        else:
            self.mtl = mtl
            if text_feat_cache_path is None:
                text_feat_cache_path = os.path.expanduser("~/.cache/pyiqa/liqe_text_feat.pt")

        if os.path.exists(text_feat_cache_path):
            self.text_features = torch.load(text_feat_cache_path, map_location='cpu')
        else:
            print(f'Generating text features for LIQE model, will be cached at {text_feat_cache_path}.')
            if self.mtl:
                self.joint_texts = torch.cat(
                    [clip.tokenize(f"a photo of a {c} with {d} artifacts, which is of {q} quality") for q, c, d
                     in product(qualitys, scenes, dists_map)])
            else:
                self.joint_texts = torch.cat([clip.tokenize(f"a photo with {c} quality") for c in qualitys])

            self.text_features = self.get_text_features(self.joint_texts)
            torch.save(self.text_features.to('cpu'), text_feat_cache_path)
    
    def get_text_features(self, x):
        text_features = self.clip_model.encode_text(self.joint_texts.to(x.device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def forward(self, x, only_score=False):              
        bs = x.size(0)
        h = x.size(2)
        w = x.size(3)

        # assert (h >= 224) & (w >= 224), 'Short side is less than 224, try upsampling the original image'
        assert (h >= 224) & (w >= 224), print(x.shape)
        # preprocess image
        x = (x - self.default_mean.to(x)) / self.default_std.to(x)

        x = x.unfold(2, 224, self.step).unfold(3, 224, self.step).permute(0, 2, 3, 1, 4, 5).reshape(bs, -1, 3, 224, 224)

        if x.size(1) < self.num_patch:
            num_patch = x.size(1)
            self.num_patch = num_patch
        else:
            num_patch = self.num_patch

        if self.training:
            sel = torch.randint(low=0, high=x.size(0), size=(num_patch, ))
        else:
            sel_step = max(1, x.size(1) // self.num_patch)
            sel = torch.zeros(num_patch)
            for i in range(num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        x = x[:, sel, ...]
        x = x.reshape(bs, num_patch, x.shape[2], x.shape[3], x.shape[4])

        text_features = self.text_features.to(x)
        x = x.view(bs*x.size(1), x.size(2), x.size(3), x.size(4))
        image_features = self.clip_model.encode_image(x, pos_embedding=True)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
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
            logits_quality = logits_per_image.sum(3).sum(2)
            quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
            return quality
                        
        if self.mtl:
            logits_per_image = logits_per_image.view(-1, len(qualitys), len(scenes), len(dists_map))
            
            logits_quality = logits_per_image.sum(3).sum(2)
            similarity_scene = logits_per_image.sum(3).sum(1)
            similarity_distortion = logits_per_image.sum(1).sum(1)
            
            distortion_index = similarity_distortion.argmax(dim=1)
            scene_index = similarity_scene.argmax(dim=1)
            quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                        4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]

            pred = 'A photo of {} with {} artifacts, which has a perceptual quality of {}'.format(scenes[scene_index], dists_map[distortion_index], quality.item())
            return quality, pred
        else:
            logits_per_image = logits_per_image.view(-1, len(qualitys))
            logits_quality = logits_per_image

            quality = 1 * logits_quality[:, 0] + 2 * logits_quality[:, 1] + 3 * logits_quality[:, 2] + \
                                4 * logits_quality[:, 3] + 5 * logits_quality[:, 4]
                            
            return quality, 'A photo has a perceptual quality of {}'.format(quality.item())

        
if __name__ == "__main__":
    model = LIQE(
        pretrained='mix',
        pretrained_model_path='./weights/liqe_mix.pt',
        text_feat_cache_path='./weights/liqe_text_feat_mix.pt'
    ).eval()

    save_name = './assert/1_雕塑.jpg'
    img = Image.open(save_name).convert('RGB')
    img = resize_img(img)
    img = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        quality, pred = model(img)
        print(quality, pred)

