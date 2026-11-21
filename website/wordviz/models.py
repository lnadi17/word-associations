"""Dataset identifiers used by the cleaned trials, graphs, and archived matrices."""

GRAPH_MODELS = {
    "humans": "Humans",
    "gemma": "Gemma",
    "gpt5_mini": "GPT-5 mini",
    "haiku": "Haiku",
    "llama3": "Llama3",
    "mistral": "Mistral",
    "kona2_12b": "Kona2 12B",
    "llama3_1_8b": "Llama3.1 8B",
    "llama3_2_3b": "Llama3.2 3B",
    "qwen2_5_7b": "Qwen2.5 7B",
}

MATRIX_MODELS = {
    stem: GRAPH_MODELS[stem.lower()]
    for stem in ("Humans", "Gpt5_mini", "Gemma", "Haiku", "Llama3", "Mistral")
}
