# Parameter-Isolated Experts for the 11th ABAW Multi-Task Challenge

Solution code for our submission [(Link)](https://arxiv.org/abs/2607.16290) to the **Multi-Task Learning (MTL) track of the
11th ABAW Competition (ECCV 2026)** on s-Aff-Wild2: joint valence–arousal
estimation, 8-class expression recognition, and 12-way action-unit detection from
a single cropped face. 

## Installation

Python 3.13.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          
```

## Reproducing the system

The pipeline runs in the frozen-feature regime: backbones are adapted per-frame,
then cached as features on which lightweight temporal heads train. Stages
(`configs/` holds one YAML per experiment; runs land under `runs/`):

1. **Affect-supervised backbones.** Fine-tune each face backbone (FSFM, MAE-Face)
   on AffectNet, then adapt per-frame on s-Aff-Wild2
   (`python -m src.engine.train --config configs/<backbone>_anft.yaml --out runs/<...>`).
   The parameter-isolated experts are low-rank (LoRA) adapters on a frozen base,
   one per affect corpus (`configs/lora_{an,fsfm,mae}.yaml`).
2. **Cache features.** `python -m src.engine.extract_features --run runs/<...>/seed_0 --out runs/<...>/feats`
   for train and validation frames (and, at test time, `--data-config` over the test list).
3. **Temporal heads.** Train BiGRU specialists (VA, AU) and the shared
   affect-latent expression head over the cached features
   (`python -m src.engine.train_temporal ...`, `--latent --zdim 96` for the
   shared-latent model).
4. **Per-task-best assembly + submission.**
   `python -m report.make_submission --split val --expr 3bb` prints the validation
   `P_MTL` and writes the submission file; `scripts/make_test_submission.sh <test_list>`
   runs the same assembly over the test frames.


## Citation

If you use this code, please cite our [pre-print](https://arxiv.org/abs/2607.16290) and the ABAW challenge:

```bibtex
@misc{cprABAW11thmtl,
  title  = {Strength-Parity Ensembling with Parameter-Isolated Experts for Multi-Task Affect Recognition},
  author = {Bui, Tung Hung and Nguyen, Hong Hai and Huynh, Van Thong},
  year   = {2026},
  note   = {arXiv pre-print}
  url= {https://arxiv.org/abs/2607.16290},
}
```

Please also cite the ABAW challenge papers as required by the organizers (see the
dataset README distributed with s-Aff-Wild2).
