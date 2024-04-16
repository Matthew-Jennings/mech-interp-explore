import re

from jaxtyping import Float
import torch as t
from transformer_lens import HookedTransformer

PATTERNS = {
    "Space, First Char Uppercase": re.compile(r"\s[A-Z]\w*"),
    "Space, First Char Lowercase": re.compile(r"\s[a-z]\w*"),
    "Space, First Char Numeral": re.compile(r"\s[0-9]\w*"),
    "No Space, First Char Uppercase": re.compile(r"[A-Z]\w*"),
    "No Space, First Char Lowercase": re.compile(r"[a-z]\w*"),
    "No Space, First Char Numeral": re.compile(r"[0-9]\w*"),
}


def cumul_probs_by_capitalisation_type(
    logits: Float[t.Tensor, "d_vocab"], model: HookedTransformer, batch=0, print_=False
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

    probs = logits[batch].softmax(-1).cpu().numpy()
    str_tokens = model.to_str_tokens(t.arange(model.cfg.d_vocab))

    cumul_probs = dict.fromkeys(PATTERNS, 0)
    cumul_probs["Other"] = 0

    for i, str_token in enumerate(str_tokens):
        matched = False
        for pattern_name, pattern in PATTERNS.items():
            if pattern.match(str_token):
                cumul_probs[pattern_name] += probs[i]
                matched = True
        if not matched:
            cumul_probs["Other"] += probs[i]

    if print_:
        for pattern, cumul_prob in sorted(
            cumul_probs.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"{pattern}: {100*cumul_prob:.2f}%")

    return cumul_probs
