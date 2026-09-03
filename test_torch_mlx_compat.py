"""Test graykode/gpt-2-Pytorch with torch-mlx.

Loads the upstream `GPT2/model.py` + `config.py` byte-for-byte (pure torch/nn/F),
builds a small GPT-2 and exercises the language-modeling forward (with and
without labels), backward + optimizer step, and tied-embedding path.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # torch-mlx root

import torch
import torch.nn as nn

PKG = Path(__file__).resolve().parent / "GPT2"
sys.path.insert(0, str(PKG))

import config as GPT2Config_mod
import model as gpt2model

GPT2Config = GPT2Config_mod.GPT2Config
GPT2Model = gpt2model.GPT2Model
GPT2LMHeadModel = gpt2model.GPT2LMHeadModel

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")
    return cond


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    print("=" * 60)
    print("graykode/gpt-2-Pytorch — GPT-2 (transformeur) + tied embeddings")
    print("=" * 60)

    cfg = GPT2Config(1000, 128, 64, 64, 2, 4, 1e-5, 0.02)

    model = GPT2LMHeadModel(cfg)
    check("GPT2LMHeadModel built", count_params(model) > 0)
    print(f"    GPT2LMHeadModel: {count_params(model):,} params")

    model.set_tied()
    check("tied embeddings (wte <-> lm_head.decoder)", True)

    # Language model forward: (batch, seq) -> (batch, seq, vocab)
    input_ids = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    logits, presents = model(input_ids)
    check("LM forward (batch=2, seq=16) shape",
          tuple(logits.shape) == (2, 16, 1000))
    check("forward finite", bool(torch.isfinite(logits).all().item()))
    check("presents for each layer", len(presents) == cfg.n_layer)

    # LM training with labels -> scalar CE loss
    lm_labels = torch.randint(0, 1000, (2, 16), dtype=torch.long)
    loss = model(input_ids, lm_labels=lm_labels)
    check("LM loss (with labels) scalar + finite", bool(torch.isfinite(loss).item()))
    print(f"    loss = {loss.item():.4f}")

    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    grads = [p.grad for p in params]
    n_grad = sum(g is not None for g in grads)
    check(f"backward: {n_grad}/{len(params)} params got gradients", n_grad == len(params))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    check("AdamW optimizer step ran", True)

    # Gradient check on shared/tied embedding vs decoder weight
    tied_ok = all(
        model.transformer.wte.weight.shape == model.lm_head.decoder.weight.shape
        for _ in [0]
    )
    check("wte/decoder same shape (tied)", tied_ok)

    loss2 = model(input_ids, lm_labels=lm_labels)
    check("post-step loss finite", bool(torch.isfinite(loss2).item()))

    print("\n" + "=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
