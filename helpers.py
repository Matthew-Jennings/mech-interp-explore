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


def correct_incorrect_answers_for_top_spacetitleword(
    logits_idx_sorted, model, topk_to_search=100
):
    """Generate a tuple of correct/incorrect answers by sampling the
    first '<space><titleword>' token in `logits_sorted_idx` and finding
    the corresponding incorrect versions. I.e.,
    '<titleword>' (no space), '<space><lowerword>', and '<lowerword>'
    """
    answer_str_tokens = []
    answer_tokens = []

    for logits_idx_sorted_batch in logits_idx_sorted:
        predictions_sorted = model.to_str_tokens(logits_idx_sorted_batch)

        top_spacetitleword_prediction = next(
            pred
            for pred in predictions_sorted[:topk_to_search]  # Assume in first 100
            if PATTERNS["Space, First Char Uppercase"].match(pred)
        )
        if top_spacetitleword_prediction is None:
            return ValueError("No space-titleword token found in topk predictions.")

        lowercase_counterpart_str_token = top_spacetitleword_prediction.lower()
        nospace_counterpart_str_token = top_spacetitleword_prediction.lstrip()
        nospace_lowercase_counterpart_str_token = (
            lowercase_counterpart_str_token.lstrip()
        )

        answer_str_tokens.append(
            (
                top_spacetitleword_prediction,
                lowercase_counterpart_str_token,
                nospace_counterpart_str_token,
                nospace_lowercase_counterpart_str_token,
            )
        )

    answer_tokens.append(model.to_tokens(answer_str_tokens[-1]).to(model.cfg.device))

    return answer_tokens, answer_str_tokens
