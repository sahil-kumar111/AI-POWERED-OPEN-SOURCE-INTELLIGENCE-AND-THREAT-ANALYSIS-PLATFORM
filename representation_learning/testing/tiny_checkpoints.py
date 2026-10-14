"""Build TINY, RANDOMLY-INITIALISED local checkpoints in Hugging Face format.

Purpose: exercise the REAL loading/inference code path (safetensors, config checks,
processor cross-check, torch forward) with no network and no pretrained weights.
These are NOT DINOv2/CLIP weights; their embeddings carry no semantic meaning.
Dimensions match the real variants' output dims so spec checks apply unchanged.
Requires torch + transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

DINOV2_PREPROC_JSON = {
    "crop_size": {"height": 224, "width": 224}, "do_center_crop": True, "do_convert_rgb": None,
    "do_normalize": True, "do_resize": True, "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "BitImageProcessor", "image_std": [0.229, 0.224, 0.225],
    "resample": 3, "rescale_factor": 0.00392156862745098, "size": {"shortest_edge": 256}}
CLIP_PREPROC_JSON = {
    "crop_size": 224, "do_center_crop": True, "do_convert_rgb": True, "do_normalize": True,
    "do_resize": True, "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_processor_type": "CLIPImageProcessor",
    "image_std": [0.26862954, 0.26130258, 0.27577711], "resample": 3,
    "rescale_factor": 0.00392156862745098, "size": 224}


def make_tiny_dinov2(path: Path, hidden: int = 384, seed: int = 0, preproc=None) -> Path:
    import torch
    from transformers import Dinov2Config, Dinov2Model
    torch.manual_seed(seed)
    cfg = Dinov2Config(hidden_size=hidden, num_hidden_layers=1, num_attention_heads=6,
                       mlp_ratio=2, image_size=224, patch_size=14)
    Dinov2Model(cfg).eval().save_pretrained(path)
    (path / "preprocessor_config.json").write_text(json.dumps(preproc or DINOV2_PREPROC_JSON))
    return path


def make_tiny_clip(path: Path, proj: int = 512, seed: int = 0, preproc=None) -> Path:
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
    torch.manual_seed(seed)
    text = CLIPTextConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                          intermediate_size=64, vocab_size=49408, max_position_embeddings=16,
                          projection_dim=proj)
    vis = CLIPVisionConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=128, image_size=224, patch_size=32,
                           projection_dim=proj)
    cfg = CLIPConfig(text_config=text.to_dict(), vision_config=vis.to_dict(),
                     projection_dim=proj)
    CLIPModel(cfg).eval().save_pretrained(path)
    (path / "preprocessor_config.json").write_text(json.dumps(preproc or CLIP_PREPROC_JSON))
    return path
