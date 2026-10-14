"""CLIP image encoder (vision tower + projection only; the text tower is NOT used).

Extracted feature: the projected image embedding (`image_embeds`, i.e. the vector
CLIP's contrastive objective aligns with CLIP *text* embeddings), L2-normalised.
Phase 2 never touches the text tower. This vector is not comparable with DINOv2
vectors, and CLIP image-text alignment has not been validated on this project's data.

Default variant: openai/clip-vit-base-patch32 (ViT-B/32, 512-d, MIT per model card).
"""

from __future__ import annotations

from typing import Optional

from .base import VisualEncoder
from .datatypes import EncoderConfig, EncoderSpec
from .preprocessing import CLIP_PREPROCESS

# variant -> (hf id, projection dim). Only 224px variants (the preprocessing is fixed).
CLIP_VARIANTS = {
    "vit-b-32": ("openai/clip-vit-base-patch32", 512),
    "vit-b-16": ("openai/clip-vit-base-patch16", 512),
    "vit-l-14": ("openai/clip-vit-large-patch14", 768),
}
FEATURE = "projected image embedding (CLIPVisionModelWithProjection.image_embeds)"


def make_spec(variant: str = "vit-b-32") -> EncoderSpec:
    if variant not in CLIP_VARIANTS:
        raise ValueError(f"unknown CLIP variant {variant!r}; choose {list(CLIP_VARIANTS)}")
    hf_id, dim = CLIP_VARIANTS[variant]
    return EncoderSpec(
        name=f"clip-{variant}", family="CLIP", feature_name=FEATURE, expected_dim=dim,
        preprocess=CLIP_PREPROCESS, license="MIT (verify on the model card)",
        source_ref=f"https://huggingface.co/{hf_id}")


class ClipEncoder(VisualEncoder):
    def __init__(self, variant: str = "vit-b-32", config: Optional[EncoderConfig] = None,
                 backend=None):
        super().__init__(make_spec(variant), config, backend)
        self.variant = variant

    def _make_backend(self):
        from .torch_backends import TorchBackend

        def load_model(tf, path):
            return tf.CLIPVisionModelWithProjection.from_pretrained(path, local_files_only=True)

        def extract(model, x):
            return model(pixel_values=x).image_embeds

        return TorchBackend(
            self.config, self.spec.preprocess, model_type="clip",
            expected_dim=self.spec.expected_dim, load_model=load_model, extract=extract,
            dim_from_config=lambda c: int(c["projection_dim"]))
