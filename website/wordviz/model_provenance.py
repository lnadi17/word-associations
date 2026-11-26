"""Evidence for the identity of each collection. Keys for served data match GRAPH_MODELS."""

from datetime import datetime


# Pin evidence to trees containing the original responses and collection scripts.
REPO = "https://github.com/lnadi17/word-associations"
TREE = "43d134537bcd48614255a80ac3a8b61583414b46"
RUN_TREE = "a7a5bc1fb2273b45fad9b43d2e6d9e7131a336c7"
LWOW = "https://github.com/LLMWorldOfWords/LWOW"
LWOW_TREE = "455ca9d475b4c3a55d48952689ba6db18cb82574"
LWOW_PAPER = ("LWOW collection methods", "https://doi.org/10.1038/s41597-025-05156-9")
UNPINNED = "The collection records do not identify an immutable weights revision."


def file_source(label, path):
    return label, f"{REPO}/blob/{TREE}/{path}"


def lwow_table(model):
    return "Authors’ archived trial table", f"{LWOW}/blob/{LWOW_TREE}/LWOW_datasets/processed_datasets/FA_{model}.csv"


def openai_collection(size):
    stem = f"gpt5_{size}"
    return {
        "label": f"GPT-5 {size} · 2025-08-07", "identifier_label": "Requested API snapshot",
        "identifier": f"gpt-5-{size}-2025-08-07",
        "revision": "2025-08-07, the dated OpenAI API snapshot in the generation configuration.",
        "collection": "Local English free-association collection using OpenAI’s batch API.",
        "sources": (("Original generation configuration",
                     f"{REPO}/blob/03a43268d0ef41638ed36522ca571c983f846f27/configs/lwow_{stem}.yaml#L15-L19"),
                    file_source("Raw response table", f"data/raw/{stem}_responses.csv"),
                    ("OpenAI model snapshots", f"https://developers.openai.com/api/docs/models/gpt-5-{size}")),
    }


def lag_collection(label, identifier, config, notes, script=None):
    sources = [file_source("Response file", f"LAG/lag_output/lag_results_{config}.csv"),
               file_source("Processing configuration", f"configs/lwow_lag_{config}.yaml")]
    if script:
        sources.append(file_source("Generation script", script))
    return {
        "label": label, "identifier": identifier,
        "identifier_label": "Configured checkpoint" if script else "Recorded dataset label",
        "revision": UNPINNED if script else "The release label and size are recorded. The checkpoint repository, tuning edition and weights revision are unrecorded.",
        "collection": "Local LAG response table, with separate cue and trial fields.",
        "notes": notes, "sources": tuple(sources),
    }


def vllm_run(label, identifier, started, config_line, audit=None):
    # The runner names the directory from the model's final path segment and its local clock.
    folder = identifier.rsplit("/", 1)[1] + "_" + started
    timestamp = datetime.strptime(started, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return {
        "label": label, "identifier": identifier, "identifier_label": "Configured vLLM checkpoint",
        "revision": "The run records the checkpoint name. The weights commit and installed vLLM version are unrecorded.",
        "collection": f"vLLM generation run, {timestamp} on the runner’s local clock. "
                      "The output directory records the start time and contains 48 response shards.",
        "availability": "Raw response batches only",
        "details": (("Raw-data check", audit),) if audit else (),
        "sources": (("Checkpoint configuration", f"{REPO}/blob/{RUN_TREE}/LAG/helper/configs.py#L{config_line}"),
                    ("Run files", f"{REPO}/tree/{RUN_TREE}/LAG/outputs/{folder}/outputs"),
                    ("Generation and output format", f"{REPO}/blob/{RUN_TREE}/LAG/vllm_runner.py#L39-L78")),
    }


MODEL_PROVENANCE = {
    "humans": {
        "label": "Humans", "identifier": "SWOW-EN18 R100",
        "revision": "English release file dated 2018-08-27.",
        "collection": "The local cleaning of the R100 human-participant file produced the 1,160,300 trials used by Explore.",
        "sources": (file_source("R100 input and cleaning method", "scripts/build_swow_r100_fa_edgelist.py#L73-L108"),
                    file_source("Processed human trials used by Explore", "data/processed/humans_processed.csv"),
                    ("SWOW study", "https://doi.org/10.3758/s13428-018-1115-7")),
    },
    "gemma": {
        "label": "Gemma · checkpoint unrecorded", "identifier_label": "Archived collection label", "identifier": "gemma",
        "revision": "The original trial table records cues and responses. The processing configuration identifies "
                    "the Gemma family. The specific release, size and tuning edition remain unknown.",
        "collection": "Imported from per-cue text files on March 1, 2026. The later Gemma 4 collections "
                      "have separate records with their configured checkpoints.",
        "sources": (("Processing configuration",
                     f"{REPO}/blob/cf3f13469ff1485afc42e7b886c9730fed438f0d/configs/lwow_gemma.yaml#L16"),
                    ("Original cue-folder importer",
                     f"{REPO}/blob/3d932c2c39b3bcb3c21f3fead1868b07c9e20d75/scripts/gemma_outputs_to_csv.py#L21-L24"),
                    ("Original data import", f"{REPO}/commit/3d932c2c39b3bcb3c21f3fead1868b07c9e20d75")),
    },
    "gpt5_mini": openai_collection("mini"),
    "haiku": {
        "label": "Claude 3.5 Haiku · 2024-10-22", "identifier_label": "Published API alias",
        "identifier": "claude-3-5-haiku-latest", "snapshot": "claude-3-5-haiku-20241022",
        "revision": "The authors report the API alias. The date in the display name comes from Anthropic’s "
                    "documented alias mapping. The released responses omit per-request model IDs.",
        "collection": "Anthropic API collection published by the LWOW authors.",
        "sources": (LWOW_PAPER, lwow_table("Haiku"),
                    ("Anthropic’s alias reference", "https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide")),
    },
    "llama3": {
        "label": "Llama3", "identifier_label": "Published model label", "identifier": "llama3.1-8b",
        "revision": "Meta Llama 3.1, 8B parameters. The authors’ reported label leaves the weights revision and tuning edition unspecified.",
        "collection": "LWOW authors’ published Llama responses. The local Llama3.1 8B collection has a separate entry.",
        "sources": (LWOW_PAPER, lwow_table("Llama3")),
    },
    "mistral": {
        "label": "Mistral 7B Instruct v0.1 · Q4_K_M", "identifier_label": "Authors’ configured model file",
        "identifier": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "revision": "Mistral 7B Instruct v0.1, Q4_K_M quantization in GGUF format. A weights checksum is unrecorded.",
        "collection": "LWOW authors’ published Mistral responses. Their generation script supplies the model filename.",
        "sources": (LWOW_PAPER, lwow_table("Mistral"),
                    ("Authors’ generation script", f"{LWOW}/blob/65b098ff3a6f6182dcbc718f5e157cb10e355700/reproducibility/FA_data_Generation.py#L7-L14")),
    },
    "kona2_12b": lag_collection("Kona2 12B", "tbilisi-ai-lab/kona2-12B-Instruct", "Kona2-12B",
        "The local script configures this model and loads it in bfloat16. Its generic output filename "
        "differs from the archived Kona file, whose rows do not include a model ID.", "LAG/lag.py#L11-L42"),
    "llama3_1_8b": lag_collection("Llama3.1 8B", "meta-llama/Llama-3.1-8B-Instruct", "Llama3.1-8B",
        "The local generation script requests the Instruct edition in bfloat16. This is a separate "
        "collection from the LWOW archive labeled Llama3.", "LAG/lag_llama.py#L11-L42"),
    "llama3_2_3b": lag_collection("Llama3.2 3B", "Llama3.2-3B", "Llama3.2-3B",
        "The raw filename and processing configuration identify the Llama 3.2 family and 3B size."),
    "qwen2_5_7b": lag_collection("Qwen2.5 7B", "Qwen2.5-7B", "Qwen2.5-7B",
        "The raw filename and processing configuration identify the Qwen 2.5 family and 7B size."),
    "gemma4_26b_uncensored_fp8": vllm_run("Gemma 4 26B-A4B IT · Uncensored · OrcaRouter FP8",
        "orcarouter/Gemma-4-26B-A4B-it-Uncensored-FP8", "20261113_175503", 15,
        "1,221,600 rows covering 12,216 cues at 100 samples each. All response fields contain text. "
        "501 generations ended at the length limit."),
    "qwen38_27b_uncensored_fp8": vllm_run("Qwen 3.8 27B · Uncensored · OrcaRouter FP8",
        "orcarouter/Qwen3.8-27B-Uncensored-FP8", "20261113_190158", 20,
        "1,221,600 rows covering 12,216 cues at 100 samples each. 2,416 response fields are blank "
        "after trimming whitespace. Two generations ended at the length limit."),
    "gemma4_12b": vllm_run("Gemma 4 12B IT", "google/gemma-4-12B-it", "20260827_212711", 12),
    "gemma4_26b": vllm_run("Gemma 4 26B-A4B IT", "google/gemma-4-26B-A4B-it", "20260827_222703", 13),
    "gemma4_e2b": vllm_run("Gemma 4 E2B IT", "google/gemma-4-E2B-it", "20260827_231932", 10),
    "gemma4_e4b": vllm_run("Gemma 4 E4B IT", "google/gemma-4-E4B-it", "20260827_233314", 11),
    "qwen38_27b_fp8": vllm_run("Qwen 3.8 27B · FP8", "Qwen/Qwen3.8-27B-FP8", "20260828_181947", 19),
    "nemotron35_lightning30b": vllm_run("Nemotron 3.5 Lightning 30B-A3B · BF16",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "20260828_005222", 3),
    "nemotron3_nano4b": vllm_run("Nemotron 3 Nano 4B · BF16",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "20260828_112926", 6),
    "nemotron3_nano30b_vllm": vllm_run("Nemotron 3 Nano 30B-A3B · BF16, vLLM",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "20260829_141657", 7),
    "nemotron3_nano30b_transformers": {
        "label": "Nemotron 3 Nano 30B-A3B · BF16, Transformers",
        "identifier_label": "Configured checkpoint", "identifier": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "revision": UNPINNED, "availability": "Raw cue and trial table only",
        "collection": "Separate Hugging Face Transformers collection with three response columns per trial.",
        "sources": (file_source("Transformers generation script", "LAG/nemotron.py#L10-L45"),
                    file_source("Raw trial table", "LAG/lag_output/lag_results_Nemotron-3-Nano-30B-A3B-BF16.csv")),
    },
    "gpt5_nano": {
        **openai_collection("nano"), "availability": "Partial raw collection only",
        "details": (("Released raw table", "334,999 trials covering 3,350 of the 12,216 planned cues."),),
    },
}
