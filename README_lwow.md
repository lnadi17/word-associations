# LWOW Claude Word Associations

This pipeline collects free-association data from Claude for comparison with the Small World of Words (SWOW) dataset. It samples SWOW cues, queries Claude for 3 associations per cue across repeated trials, applies minimal normalization, and analyzes association strength vs embedding distance using local word embeddings.

## Overview

The protocol mirrors the LWOW methodology: each cue word is presented multiple times; the model responds with exactly 3 comma-separated associate words; responses are minimally processed; and association strength (edge frequency) is correlated with embedding distance to assess semantic alignment.

## Project Structure

```
lwow/
  config.py          # Configuration dataclasses
  io.py              # CSV read/write helpers
  sampling.py        # Cue sampling from SWOW
  generation.py      # Claude prompting and response parsing
  processing.py      # Response normalization and edge counting
  analysis.py        # Embedding distances and correlation
  clients/
    anthropic_client.py  # Claude Messages API client
    embeddings.py        # Local (gensim) or remote (Voyage) embeddings

scripts/
  generate_associations.py   # Step 1: call Claude
  process_associations.py   # Step 2: normalize and build edges
  analyze_embeddings.py     # Step 3: correlate counts with distances

configs/
  lwow_claude.yaml   # YAML configuration

data/
  raw/               # Raw API responses
  processed/         # Normalized trials and edge list
  analysis/          # Correlation stats and outliers
```

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

Required packages: `anthropic`, `gensim`, `numpy`, `pyyaml`, `requests`, `scipy`, `tqdm`.

### Environment

For generation (required):

```bash
export ANTHROPIC_API_KEY="..."
```

For optional Voyage embeddings (if not using local gensim):

```bash
export VOYAGE_API_KEY="..."
```

### PYTHONPATH

Run scripts from the project root with the `lwow` package on the path:

```bash
PYTHONPATH=. python3 scripts/generate_associations.py --config configs/lwow_claude.yaml
```

## Configuration

Edit `configs/lwow_claude.yaml`. Key options:


| Section    | Key                   | Description                                    | Default                                                |
| ---------- | --------------------- | ---------------------------------------------- | ------------------------------------------------------ |
| paths      | cue_stats_path        | Path to SWOW cue stats CSV                     | `lexicon/SWOW-EN18/cueStats.SWOW-EN.R123.20180827.csv` |
| paths      | raw_output_path       | Raw responses output                           | `data/raw/claude_opus45_responses.csv`                 |
| paths      | edge_list_output_path | Edge list (src, tgt, wt)                       | `data/processed/claude_opus45_edges.csv`               |
| sampling   | num_cues              | Number of cues to sample                       | 1000                                                   |
| sampling   | seed                  | Random seed for sampling                       | 42                                                     |
| generation | model                 | Claude model ID                                | `claude-opus-4-5`                                      |
| generation | repetitions_per_cue   | Trials per cue                                 | 100                                                    |
| generation | max_tokens            | Max output tokens                              | 32                                                     |
| generation | temperature           | Sampling temperature                           | 0.7                                                    |
| generation | max_retries           | Retries on parse failure                       | 3                                                      |
| processing | repetitions_per_cue   | Must match generation                          | 100                                                    |
| processing | seed                  | For down-sampling duplicates                   | 42                                                     |
| embeddings | provider              | `gensim` or `word2vec` or `voyage`             | `gensim`                                               |
| embeddings | model_name            | Gensim model (e.g. `glove-wiki-gigaword-100`)  | `glove-wiki-gigaword-100`                              |
| embeddings | model_path            | Path to word2vec file (when provider=word2vec) | —                                                      |
| embeddings | oov_strategy          | `skip` or `zero` for unknown words             | `skip`                                                 |
| analysis   | count_percentile      | Percentile for high-frequency edges            | 0.9                                                    |
| analysis   | distance_percentile   | Percentile for high-distance outliers          | 0.9                                                    |


## Protocol (Step by Step)

### Step 1: Generate Raw Associations

**Script:** `scripts/generate_associations.py`

1. Load cues from `cue_stats_path` (CSV with `cue` column); sample `num_cues` cues using `sampling.seed`.
2. For each cue, send `repetitions_per_cue` prompts to Claude.
3. **Prompt format:**
  - System: fixed task instructions.  
  - User: `Input: <cue>\nOutput:`  
  - Expected response: exactly 3 words separated by commas (e.g. `water, beach, sun`).
4. Parse responses: strip punctuation, split on comma, take first 3 tokens. If fewer than 3, retry up to `max_retries` with exponential backoff.
5. Write one row per (cue, trial) with columns: `cue`, `trial`, `raw_text`, `parsed` (pipe-separated), `ok` (True if 3 words parsed).

**Fixed system prompt (from config):**

```
Task:

• You will be provided with an input word: write the first 3 words you associate to it separated by a comma
• No additional output text is allowed  Constraints:
• No carriage return characters are allowed in the answers
• Answers should be as short as possible

Example: 
Input: sea
Output: water, beach, sun
```

### Step 2: Process and Build Edge List

**Script:** `scripts/process_associations.py`

1. Read raw output CSV.
2. **Normalization:** lowercase; replace underscores with spaces; remove self-responses (response = cue); deduplicate responses within each trial.
3. **Repetition alignment:** for each cue, ensure exactly `repetitions_per_cue` trials: if more, randomly sample down; if fewer, pad with blank responses.
4. Write processed trials to `processed_output_path` (cue, trial, responses, raw_text).
5. Build edge counts: for each (cue, response) pair, count occurrences across trials.
6. Write edge list to `edge_list_output_path` with columns: `src`, `tgt`, `wt`.

### Step 3: Analyze Embedding Distance

**Script:** `scripts/analyze_embeddings.py`

1. Read edge list.
2. Collect vocabulary (all unique `src` and `tgt`).
3. With `provider=gensim`: load `model_name` via `gensim.downloader`; for each word, look up vector (try word and `word.replace(" ", "_")`); skip words not in vocabulary (`oov_strategy=skip`).
4. For each edge (src, tgt, wt), compute cosine distance between embeddings: `1 - cosine_similarity`.
5. **Correlation:** Pearson and Spearman between edge weight (`wt`) and cosine distance. Higher association strength with lower distance suggests semantic consistency.
6. **Outliers:** edges in the top `count_percentile` by weight and top `distance_percentile` by distance—high frequency but high distance (potential non-semantic associations).
7. Write `embedding_correlation.csv` (pearson_r, pearson_p, spearman_r, spearman_p, missing_words, missing_edges) and `outliers.csv` (src, tgt, wt, distance).

## Output Files


| File                                         | Description                                          |
| -------------------------------------------- | ---------------------------------------------------- |
| `data/raw/claude_opus45_responses.csv`       | Raw API responses: cue, trial, raw_text, parsed, ok  |
| `data/processed/claude_opus45_processed.csv` | Normalized trials: cue, trial, responses, raw_text   |
| `data/processed/claude_opus45_edges.csv`     | Edge list: src, tgt, wt (frequency)                  |
| `data/analysis/embedding_correlation.csv`    | Pearson/Spearman stats, missing_words, missing_edges |
| `data/analysis/outliers.csv`                 | High-weight, high-distance edge pairs                |


## Run Commands

```bash
# Step 1: Generate (requires ANTHROPIC_API_KEY)
PYTHONPATH=. python3 scripts/generate_associations.py --config configs/lwow_claude.yaml

# Step 2: Process
PYTHONPATH=. python3 scripts/process_associations.py --config configs/lwow_claude.yaml

# Step 3: Analyze embeddings (downloads gensim model on first run)
PYTHONPATH=. python3 scripts/analyze_embeddings.py --config configs/lwow_claude.yaml
```

## Cost Estimate (Claude Opus 4.5)

For ~12,000 cues × 100 repetitions = 1.2M requests (Anthropic pricing: $5/MTok input, $25/MTok output):

- Input: 90 tokens/request × 1.2M ≈ 108M tokens → **$540**
- Output: 6 tokens/request × 1.2M ≈ 7.2M tokens → **$180**
- **Total: ~$700–800** (approximate)

Batch processing or prompt caching can reduce this.