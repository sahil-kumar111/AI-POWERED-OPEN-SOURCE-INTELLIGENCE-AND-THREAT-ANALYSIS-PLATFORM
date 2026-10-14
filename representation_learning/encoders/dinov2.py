"""DINOv2 image encoder (self-supervised ViT; image-only, no text tower).

Extracted feature: the CLS token of the final layer after the final LayerNorm
(`Dinov2Model.pooler_output`), dim = hidden size of the variant. Patch tokens are
not used. L2-normalised after extraction.

Default variant: facebook/dinov2-small (ViT-S/14, 384-d) - chosen for CPU feasibility, NOT as
a quality ranking. `variant="base"` (768-d) etc. remain selectable.
"""

from __future__ import annotations

from typing import Optional

from .base import VisualEncoder
from .datatypes import EncoderConfig, EncoderSpec
from .preprocessing import DINOV2_PREPROCESS

# variant -> (hf id, dim)
DINOV2_VARIANTS = {
    "small": ("facebook/dinov2-small", 384),
    "base": ("facebook/dinov2-base", 768),
    "large": ("facebook/dinov2-large", 1024),
    "giant": ("facebook/dinov2-giant", 1536),
}
FEATURE = "CLS token, final layer, after final LayerNorm (Dinov2Model.pooler_output)"


def make_spec(variant: str = "small") -> EncoderSpec:
    if variant not in DINOV2_VARIANTS:
        raise ValueError(f"unknown DINOv2 variant {variant!r}; choose {list(DINOV2_VARIANTS)}")
    hf_id, dim = DINOV2_VARIANTS[variant]
    return EncoderSpec(
        name=f"dinov2-{variant}", family="DINOv2", feature_name=FEATURE, expected_dim=dim,
        preprocess=DINOV2_PREPROCESS, license="Apache-2.0 (verify on the model card)",
        source_ref=f"https://huggingface.co/{hf_id}")


class Dinov2Encoder(VisualEncoder):
    def __init__(self, variant: str = "small", config: Optional[EncoderConfig] = None,
                 backend=None):
        super().__init__(make_spec(variant), config, backend)
        self.variant = variant

    def _make_backend(self):
        from .torch_backends import TorchBackend

        def load_model(tf, path):
            return tf.Dinov2Model.from_pretrained(path, local_files_only=True)

        def extract(model, x):
            out = model(pixel_values=x)
            feat = getattr(out, "pooler_output", None)
            return feat if feat is not None else out.last_hidden_state[:, 0]

        return TorchBackend(
            self.config, self.spec.preprocess, model_type="dinov2",
            expected_dim=self.spec.expected_dim, load_model=load_model, extract=extract,
            dim_from_config=lambda c: int(c["hidden_size"]))
