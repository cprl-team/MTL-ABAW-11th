from __future__ import annotations

import timm

VARIANT = "vit_base_patch14_reg4_dinov2"


def build_dinov2(weights_path=None, img_size: int = 224):
    model = timm.create_model(VARIANT, pretrained=True, num_classes=0, img_size=img_size)
    return model, model.num_features
