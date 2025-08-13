#!/usr/bin/env python3
"""
LoRA fine-tuning for CosyVoice 2
=================================
- **No offset gymnastics**: We keep the original CosyVoice2 field layout and let ``Qwen2LM`` build the loss internally.
- **Single-GPU friendly**: HuggingFace Trainer + PEFT-LoRA.

**TSV manifest (4 columns)**
```
spk_id <TAB> text <TAB> token.npy <TAB> wav_path
```
Only *text* and *speech-token ids* are used for training.
"""

from __future__ import annotations

import argparse
import random
from logging import getLogger, StreamHandler, INFO
from pathlib import Path
from typing import List, Tuple

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    Trainer,
    TrainingArguments,
)
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

from scripts.cv2.patch import apply_patch

apply_patch()

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

huggingface_hub.cached_download = hf_hub_download

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


class CV2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_dict = model(inputs, self.args.device)
        loss = loss_dict["loss"]
        return (loss, loss_dict) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
    ):
        with torch.no_grad():
            outputs = model(inputs, self.args.device)

        loss = outputs["loss"]
        return (loss, None, None)


class TSVSpeechDataset(Dataset):
    """Reads 4-column TSV and returns (text, speech_ids)"""

    def __init__(self, tsv_path: str):
        self.rows: List[Tuple[str, torch.Tensor]] = []
        for ln in Path(tsv_path).read_text(encoding="utf-8").splitlines():
            _, txt, npy, wav = ln.split("\t")
            ids = torch.from_numpy(np.load(npy)).long()  # (T,)
            self.rows.append((txt, ids, wav))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn(batch, tokenizer):
    """Make inputs exactly as *Qwen2LM.forward* expects."""

    texts, speech_ids, _ = zip(*batch)
    txt_lists = [
        tokenizer.encode(t, allowed_special=tokenizer.special_tokens) for t in texts
    ]

    pad_id = tokenizer.encode(
        "<|endoftext|>", allowed_special=tokenizer.special_tokens
    )[0]
    max_len = max(len(x) for x in txt_lists)
    txt_tok = torch.full((len(txt_lists), max_len), pad_id, dtype=torch.long)

    for i, ids in enumerate(txt_lists):
        txt_tok[i, : len(ids)] = torch.tensor(ids)

    txt_len = torch.tensor([len(ids) for ids in txt_lists], dtype=torch.int32)

    sp_pad = pad_sequence(speech_ids, batch_first=True, padding_value=0)
    sp_len = torch.tensor([t.size(0) for t in speech_ids], dtype=torch.int32)

    return {
        "text_token": txt_tok,
        "text_token_len": txt_len,
        "speech_token": sp_pad,
        "speech_token_len": sp_len,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="config.yml", help="YAML file for configuration."
    )
    ap.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="path to checkpoint dir or True for auto-resume",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    seed = cfg["training"]["seed"]
    rng = torch.Generator().manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cv2 = CosyVoice2(model_dir=cfg["base_model"], fp16=False)
    base_model = cv2.model.llm

    # Expand vocabulary
    tok = get_qwen_tokenizer(
        token_path=f"{cfg['base_model']}/CosyVoice-BlankEN", skip_special_tokens=True
    )

    # Register new special tokens
    new_tokens = ["<PHON_START>", "<PHON_END>"]
    added = tok.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    logger.info("Number of tokens added: %s", added)

    # Update the meta information on the QwenTokenizer
    tok.special_tokens["additional_special_tokens"].extend(
        [
            t
            for t in new_tokens
            if t not in tok.special_tokens["additional_special_tokens"]
        ]
    )
    base_model.llm.model.resize_token_embeddings(len(tok.tokenizer))

    lora_cfg = LoraConfig(
        r=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
    )
    llm = get_peft_model(base_model, lora_cfg)

    new_ids = tok.tokenizer.convert_tokens_to_ids(new_tokens)

    # Unfreeze embed_tokens
    emb = llm.base_model.llm.model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    # Register a hook to reset the gradient of existing tokens
    existing_mask = torch.ones(
        emb.num_embeddings, dtype=torch.bool, device=emb.weight.device
    )
    existing_mask[new_ids] = False  # False = leave gradient

    cv2.model.llm = llm
    cv2.frontend.tokenizer = tok

    # Dataset split
    ds_full = TSVSpeechDataset(cfg["manifest"])
    n = len(ds_full)
    n_val = int(n * cfg["val_ratio"])
    n_train = n - n_val
    train_ds, val_ds = random_split(ds_full, [n_train, n_val], generator=rng)

    trainer = CV2Trainer(
        model=llm,
        args=TrainingArguments(**cfg["training"]),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, tok),
    )

    def mask_grad(grad):
        if not hasattr(mask_grad, "done"):
            logger.debug("=== GradDebug (hook) ===")
            logger.debug("grad.shape :", tuple(grad.shape))
            logger.debug("new_tok_grad :", grad[new_ids])
            logger.debug("old_tok_grad(sample):", grad[0][:4], "...")
            mask_grad.done = True
        grad[existing_mask] = 0
        return grad

    emb.weight.register_hook(mask_grad)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    llm.save_pretrained(cfg["training"]["output_dir"])

    # Save additional token weights separately
    weight = emb.weight.detach().cpu()
    embed_rows = weight[new_ids]
    save_payload = {"embed_rows": embed_rows}
    save_file(
        save_payload,
        str(Path(cfg["training"]["output_dir"]) / "embed_patch.safetensors"),
    )


if __name__ == "__main__":
    main()
