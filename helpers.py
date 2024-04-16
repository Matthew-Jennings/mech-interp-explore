import re

from jaxtyping import Float
import torch as t
from transformer_lens import HookedTransformer


def cumul_probs_by_capitalisation_type(
    logits: Float[t.Tensor, "d_vocab"], model: HookedTransformer, batch=0
) -> dict:
    """
    Calculate the cumulative probability of predicting a token of each
    of the following patterns:
    - first character space, first letter uppercase
    - first character space, first letter lowercase
    - first character space, second character numeral
    - first character not a space, first letter uppercase
    - first character not a space, first letter lowercase
    - first character numeral
    - other
    """

    if batch >= logits.size()[0]:
        raise ValueError("Bad batch index.")

    patterns = {
        "Space, Upper": re.compile(r"\s[A-Z]\w*"),
        "Space, Lower": re.compile(r"\s[a-z]\w*"),
        "Space, Numeral": re.compile(r"\s[0-9]\w*"),
        "No Space, Upper": re.compile(r"[A-Z]\w*"),
        "No Space, Lower": re.compile(r"[a-z]\w*"),
        "No Space, Numeral": re.compile(r"[0-9]\w*"),
    }
    probs = logits[batch].softmax(-1).cpu().numpy()
    str_tokens = model.to_str_tokens(t.arange(model.cfg.d_vocab))

    cumul_probs = dict.fromkeys(patterns, 0)
    cumul_probs["Other"] = 0

    for i, str_token in enumerate(str_tokens):
        matched = False
        for pattern, regex in patterns.items():
            if regex.match(str_token):
                cumul_probs[pattern] += probs[i]
                matched = True
        if not matched:
            cumul_probs["Other"] += probs[i]

    return cumul_probs
