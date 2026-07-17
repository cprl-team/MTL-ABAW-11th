from __future__ import annotations

DEFAULT_TARGETS = r".*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))"


def wrap_lora(backbone, rank: int = 16, alpha: int = 32, dropout: float = 0.0, targets=None):
    """Return the backbone wrapped with peft LoRA (base frozen, adapters trainable)."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets or DEFAULT_TARGETS, bias="none")
    return get_peft_model(backbone, cfg)


def set_lora_trainable(backbone) -> tuple[int, int]:
    """Re-assert base-frozen / LoRA-trainable after any freeze schedule. Returns (n_frozen, n_train)."""
    frz = tr = 0
    for name, p in backbone.named_parameters():
        is_lora = "lora_" in name
        p.requires_grad_(is_lora)
        tr += is_lora
        frz += not is_lora
    return frz, tr
