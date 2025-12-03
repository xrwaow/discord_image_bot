import random
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from vars import (
    DEFAULT_POSITIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT,
    KEYWORDS,
    LORA_CONFIG,
    WILDCARDS,
    txt2img_args,
)

KEY_TOKEN = re.compile(r"\{([^{}]+)\}")
NUMBER_TOKEN = re.compile(r"-?\d+(?:\.\d+)?")
SUMMARY_SKIP = {"prompt", "neg_prompt", "display_prompt", "display_neg_prompt"}

SAMPLING_FIELDS: Sequence[Tuple[Sequence[str], str, callable | None]] = (
    (("cfg",), "cfg", None),
    (("sampler_name", "sampler"), "sampler", None),
    (("scheduler",), "scheduler", None),
    (("denoising_strength",), "denoise", None),
    (("scale",), "scale", None),
)
RANDOM_FIELDS: Sequence[Tuple[Sequence[str], str, callable | None]] = (
    (("seed",), "seed", None),
    (("variation_seed",), "var seed", None),
    (("batch_size",), "bs", None),
)
MODEL_FIELDS: Sequence[Tuple[Sequence[str], str, callable | None]] = (
    (("lora",), "lora", lambda value: ", ".join(_normalize_list(value))),
    (("vae",), "vae", None),
    (("clip_skip",), "clip skip", None),
)


def _clean(text: str) -> str:
    return (text or "").strip(" ,")


def _replace_keywords(text: str, cache: Dict[str, str] | None = None) -> str:
    if not text:
        return ""

    cache = {} if cache is None else cache

    def substitute(match) -> str:
        key = match.group(1).strip()
        if not key:
            return match.group(0)
        if key in cache:
            return cache[key]

        if key in KEYWORDS:
            cache[key] = _replace_keywords(KEYWORDS[key], cache)
        elif key in WILDCARDS:
            options = [option for option in WILDCARDS[key] if _clean(option)]
            selection = random.choice(options) if options else ""
            cache[key] = _replace_keywords(selection, cache)
        else:
            cache[key] = match.group(0)
        return cache[key]

    previous = None
    current = text
    while current != previous:
        previous = current
        current = KEY_TOKEN.sub(substitute, current)
    return current


def _normalize_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[|,]", value) if part.strip()]
    return [_clean(str(item)) for item in value if _clean(str(item))]


def _collect_lora_keywords(lora_names: Iterable[str] | None) -> List[str]:
    seen: set[str] = set()
    keywords: List[str] = []
    for name in lora_names or []:
        for entry in LORA_CONFIG.get(name, []):
            for word in _normalize_list(entry.get("keywords")):
                if word not in seen:
                    seen.add(word)
                    keywords.append(word)
    return keywords


def _ensure_prefix(prompt: str, addition: str, *, prepend: bool) -> str:
    addition = _clean(addition)
    if not addition:
        return prompt
    if not prompt:
        return addition
    if addition in prompt:
        return prompt
    return f"{addition}, {prompt}" if prepend else f"{prompt}, {addition}"


def preprocess_prompt(prompt: str, neg_prompt: str | None, lora_names: Iterable[str] | None = None) -> Dict[str, str]:
    token_cache: Dict[str, str] = {}

    base_prompt = _clean(prompt or "")
    display_prompt = _clean(_replace_keywords(base_prompt, token_cache))

    keywords = _collect_lora_keywords(lora_names)
    prompt_with_keywords = _ensure_prefix(base_prompt, ", ".join(keywords), prepend=True) if keywords else base_prompt
    prompt_with_defaults = _ensure_prefix(prompt_with_keywords, DEFAULT_POSITIVE_PROMPT, prepend=False)
    positive = _clean(_replace_keywords(prompt_with_defaults, token_cache))

    user_negative_raw = _clean(neg_prompt or "")
    user_negative = _clean(_replace_keywords(user_negative_raw, token_cache))
    default_negative = _clean(_replace_keywords(DEFAULT_NEGATIVE_PROMPT, token_cache)) if DEFAULT_NEGATIVE_PROMPT else ""

    if user_negative and user_negative != default_negative:
        negative = f"{user_negative}, {default_negative}" if default_negative else user_negative
    else:
        negative = default_negative
    negative = _clean(negative)

    result: Dict[str, str] = {"prompt": positive, "neg_prompt": negative, "display_prompt": display_prompt}
    if user_negative and user_negative != default_negative:
        result["display_neg_prompt"] = user_negative

    return result


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _normalize_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(part for part in (_normalize_scalar(item) for item in value) if part)
    text = _clean(str(value))
    if not text:
        return ""
    if NUMBER_TOKEN.fullmatch(text):
        try:
            return _format_number(float(text))
        except ValueError:
            pass
    return text


def _pick_value(args: Dict, keys: Sequence[str], normalizer) -> Tuple[str, str]:
    for key in keys:
        if key in args:
            value = (normalizer or _normalize_scalar)(args[key])
            if value:
                return key, value
    return "", ""


def _format_field(label: str, value: str) -> str:
    return f"**{label}**: {value};"


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    return num if num >= 0 else None


def _dimension_segment(args: Dict) -> Tuple[str, set[str]]:
    width = _coerce_int(args.get("width"))
    height = _coerce_int(args.get("height"))
    steps = _coerce_int(args.get("steps"))
    used: set[str] = set()

    if width is None and height is None:
        return "", used
    if width is not None:
        used.add("width")
    if height is not None:
        used.add("height")

    if width is not None and height is not None:
        segment = f"**{width}x{height}**"
    else:
        segment = f"**{width if width is not None else height}**"

    if steps is not None:
        used.add("steps")
        segment = f"{segment}@**{steps}**"

    return segment, used


def _collect_groups(args: Dict, fields: Sequence[Tuple[Sequence[str], str, callable | None]], consumed: set[str], skip_defaults: bool = False) -> List[str]:
    items: List[str] = []
    for keys, label, normalizer in fields:
        key, value = _pick_value(args, keys, normalizer)
        if key:
            if skip_defaults and key in txt2img_args and str(args.get(key)) == str(txt2img_args.get(key)):
                consumed.add(key)
                continue
            items.append(_format_field(label, value))
            consumed.add(key)
    return items


def format_generation_summary(gen_args: Dict, default_model: str) -> str:
    args = {key: value for key, value in gen_args.items() if value not in (None, "", [])}
    segments: List[str] = []
    consumed: set[str] = set()

    segment, used = _dimension_segment(args)
    if segment:
        segments.append(segment)
        consumed.update(used)

    for field_group in (SAMPLING_FIELDS, RANDOM_FIELDS):
        group = _collect_groups(args, field_group, consumed, skip_defaults=True)
        if group:
            segments.append(" ".join(group))

    remaining: List[str] = []
    for key, value in args.items():
        if key in consumed or key in SUMMARY_SKIP:
            continue
        text = _normalize_scalar(value)
        if text:
            remaining.append(_format_field(key.replace("_", " "), text))
    if remaining:
        segments.append(" ".join(remaining))

    model_value = _normalize_scalar(args.get("model")) or default_model
    if "model" in args and model_value:
        consumed.add("model")

    model_group = [_format_field("model", model_value)]
    model_group.extend(_collect_groups(args, MODEL_FIELDS, consumed))
    segments.append(" ".join(model_group))

    if not segments:
        segments.append(_format_field("model", default_model))

    return "> " + " | ".join(segments)