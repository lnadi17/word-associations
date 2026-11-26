"""Pin one developer release per origin using the site's lexical-decision results."""

from functools import lru_cache

from dash import html

from .experiment_data import ldt_stats, metadata
from .model_provenance import MODEL_PROVENANCE
from .models import MATRIX_MODELS, ordered_model_keys

# Collections with a documented developer API or original checkpoint.
# Dataset keys are lowercase here; each dropdown retains its own stored values.
ORIGINS = {
    "gpt5_mini": "OpenAI", "haiku": "Anthropic",
    "llama3_1_8b": "Meta",
    "qwen38_27b_fp8": "Qwen",
    "gemma4_12b": "Google", "gemma4_26b": "Google",
    "gemma4_e2b": "Google", "gemma4_e4b": "Google",
    "nemotron35_lightning30b": "NVIDIA", "nemotron3_nano4b": "NVIDIA",
    "nemotron3_nano30b_vllm": "NVIDIA", "nemotron3_nano30b_transformers": "NVIDIA",
}


@lru_cache(maxsize=1)
def pinned_models():
    if not MATRIX_MODELS:
        return {}
    experiment = metadata("ldt")
    by_origin = {}
    for model in MATRIX_MODELS:
        key = model.lower()
        origin = ORIGINS.get(key)
        if not origin or any(pair is None for pair in experiment["activation"][model]):
            continue
        rho = ldt_stats(model, experiment)[2]
        if rho is not None and rho < 0:
            candidate = (rho, key)
            if origin not in by_origin or candidate < by_origin[origin]:
                by_origin[origin] = candidate
    return {key: rho for rho, key in sorted(by_origin.values())[:5]}


def pinned_label(name):
    return html.Span([
        html.Span("★", className="model-pin", **{"aria-hidden": "true"}),
        html.Span(name, className="model-name"), html.Span(" (pinned)", className="visually-hidden"),
    ])


def model_options(models):
    pins = pinned_models()
    options = []
    for value in ordered_model_keys(models):
        key, name = value.lower(), models[value]
        record = MODEL_PROVENANCE[key]
        search = " ".join(filter(None, (name, record["identifier"], record.get("snapshot"))))
        if key in pins:
            origin = ORIGINS[key]
            options.append({"value": value, "label": pinned_label(name), "title": f"Pinned · {origin}",
                            "search": f"{search} {origin} pinned"})
        else:
            options.append({"value": value, "label": name, "search": search})
    return options
