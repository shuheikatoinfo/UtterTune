from __future__ import annotations
from typing import Callable, Any


def apply_patch():
    try:
        from submodules.CosyVoice.cosyvoice.utils import frontend_utils as fu
    except Exception as e:
        raise RuntimeError(f"[patch] failed to import cosyvoice frontend_utils: {e}")

    if getattr(fu, "_split_paragraph_patched", False):
        return

    original: Callable[..., Any] = fu.split_paragraph

    def split_paragraph_patched(
        text,
        tokenizer_encode,
        lang,
        *,
        token_max_n=80,
        token_min_n=60,
        merge_len=20,
        comma_split=False,
    ):
        if lang == "zh":
            token_max_n = 160
            merge_len = 160
        return original(
            text,
            tokenizer_encode,
            lang,
            token_max_n=token_max_n,
            token_min_n=token_min_n,
            merge_len=merge_len,
            comma_split=comma_split,
        )

    fu.split_paragraph = split_paragraph_patched
    fu._split_paragraph_patched = True
    print(
        "[patch] Patched cosyvoice.utils.frontend_utils.split_paragraph for zh "
        "(token_max_n=160, merge_len=160)"
    )
