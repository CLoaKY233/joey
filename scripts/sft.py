"""Conversation SFT (response-only masking) on DailyDialog, starting from the
base checkpoint. Saves artifacts/joey_chat.pt."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from joey.tokenizer import JoeyTokenizer
from joey.train import load_ckpt, save_ckpt
from joey.sft import build_dailydialog, run_sft


def main():
    tok = JoeyTokenizer.load("artifacts/tok.json")
    model, cfg, _ = load_ckpt("artifacts/joey_base.pt", use_ema=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_dailydialog(tok, cfg.ctx_len)
    print(f"dailydialog pairs: {len(data)}")
    run_sft(model, data, dev, steps=3000, lr=1e-4, batch_size=32)
    save_ckpt(model, cfg, "artifacts/joey_chat.pt", 3000)
    print("saved artifacts/joey_chat.pt")


if __name__ == "__main__":
    main()
