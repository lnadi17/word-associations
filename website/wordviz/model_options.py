"""Pin one developer release per origin using the site's lexical-decision results."""

from functools import lru_cache

from dash import html

from .experiment_data import ldt_stats, metadata
from .model_provenance import MODEL_PROVENANCE
from .models import MATRIX_MODELS

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


def model_options(models):
    pins = pinned_models()
    priority = {key: index for index, key in enumerate(("humans", *pins))}
    options = []
    for value in sorted(models, key=lambda key: priority.get(key.lower(), len(priority))):
        key, name = value.lower(), models[value]
        if key in pins:
            origin = ORIGINS[key]
            label = html.Span([
                html.Span("★", className="model-pin", **{"aria-hidden": "true"}),
                html.Span(name), html.Span(" (pinned)", className="visually-hidden"),
            ])
            options.append({"value": value, "label": label, "title": f"Pinned · {origin}",
                            "search": f"{name} {origin} {MODEL_PROVENANCE[key]['identifier']} pinned"})
        else:
            options.append({"value": value, "label": name})
    return options
