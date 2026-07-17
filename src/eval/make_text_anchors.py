from __future__ import annotations

import argparse

import numpy as np
import torch

EXPR_WORDS = {
    "Neutral": ["neutral", "expressionless", "with a neutral expression"],
    "Anger": ["angry", "with an angry expression", "furious"],
    "Disgust": ["disgusted", "with a disgusted expression", "showing disgust"],
    "Fear": ["fearful", "afraid", "with a frightened expression"],
    "Happiness": ["happy", "smiling", "with a happy expression"],
    "Sadness": ["sad", "with a sad expression", "unhappy"],
    "Surprise": ["surprised", "astonished", "with a surprised expression"],
    "Other": ["with another expression", "with an ambiguous expression", "other"],
}
EXPR_ORDER = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]
TEMPLATES = [
    "a photo of a {} person",
    "a face that is {}",
    "a close-up photo of a {} face",
    "a person who is {}",
]


def main():
    import open_clip
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--out", default="weights/expr_text_anchors.npy")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(args.model)

    anchors = []
    with torch.no_grad():
        for cls in EXPR_ORDER:
            prompts = [t.format(w) for w in EXPR_WORDS[cls] for t in TEMPLATES]
            emb = model.encode_text(tok(prompts).to(device)).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            a = emb.mean(0)
            a = a / a.norm()
            anchors.append(a.cpu().numpy())
    A = np.stack(anchors).astype(np.float32)
    np.save(args.out, A)
    print(f"wrote {args.out}  shape={A.shape}")
    sim = A @ A.T
    print("mean off-diagonal cosine:", round(float((sim.sum() - 8) / 56), 3))


if __name__ == "__main__":
    main()
