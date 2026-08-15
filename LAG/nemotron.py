import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline, GenerationConfig
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json
import re

# --- Configuration ---
MODEL_PATH = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
INPUT_FILE = "master_cues.json"
OUTPUT_FILE = "nemotron_results.csv"
NUM_SAMPLES = 1  # Samples per Cue
BATCH_SIZE = 8     # Reduced to prevent OOM with a 30B model
TOKENS = 30        # Answer is just "word,word,word" — thinking is disabled, no need for 250
TEMP = 1           # Temperature

# --- 1. Dataset Class for Batching ---
class PromptDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]['prompt']


def run_hpc_batch_probe():
    print(f"--- Loading Tokenizer and {MODEL_PATH} ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        device_map="auto",
        dtype=torch.bfloat16,   # torch_dtype is deprecated -> dtype
    )

    # Build a single, explicit GenerationConfig up front instead of passing
    # loose kwargs to pipe(...) every call. This avoids the "generation_config
    # together with generation-related arguments" deprecation warning and the
    # max_new_tokens/max_length mismatch warning (model default max_length=20
    # was clashing with our max_new_tokens=250).
    gen_config = GenerationConfig(
        max_new_tokens=TOKENS,
        do_sample=True,
        temperature=TEMP,
        pad_token_id=tokenizer.pad_token_id,
    )

    # --- 2. Load Cues & Calculate Missing Work ---
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, 'r') as f:
        all_cues = json.load(f)

    # Load existing progress to prevent duplicate work
    completed_tasks = set()
    if os.path.exists(OUTPUT_FILE):
        df_done = pd.read_csv(OUTPUT_FILE, usecols=['cue', 'trial_id'])
        completed_tasks = set(zip(df_done['cue'].astype(str), df_done['trial_id'].astype(int)))
        print(f"Found {len(completed_tasks)} completed trials in existing CSV.")
    else:
        # Create the file with headers
        pd.DataFrame(columns=['cue', 'trial_id', 'response_1', 'response_2', 'response_3']).to_csv(
            OUTPUT_FILE, index=False
        )

    # --- 3. Build the Task Matrix ---
    print("Building task queue...")
    tasks = []

    system_prompt = (
        "- You will be provided with an input word: write the first 3 words you associate to it separated by a comma.\n"
        "- No additional output text is allowed.\n\n"
        "Constraints:\n"
        "- no carriage return characters are allowed in the answers.\n"
        "- answers should be as short as possible.\n\n"
        "Example:\n"
        "Input: sea\n"
        "Output: water,beach,sun"
    )

    for cue in all_cues:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input: {cue}\nOutput:"}
        ]

        # Nemotron-3-Nano is a unified reasoning/non-reasoning model. By default it
        # emits a <think>...</think> trace before the final answer, which was eating
        # the entire TOKENS budget and leaving raw_response full of unfinished
        # reasoning fragments instead of the actual word associations.
        # enable_thinking=False makes it skip straight to the final answer.
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        for trial in range(1, NUM_SAMPLES + 1):
            if (cue, trial) not in completed_tasks:
                tasks.append({
                    'cue': cue,
                    'trial_id': trial,
                    'prompt': prompt
                })

    if not tasks:
        print("All trials for all cues are complete! Exiting.")
        return

    print(f"Queue built: {len(tasks)} inferences remaining.")

    # --- 4. The High-Speed Batch Inference Loop ---
    dataset = PromptDataset(tasks)
    print(f"--- Starting Batch Probing (Batch Size: {BATCH_SIZE}) ---")

    generator = pipe(
        dataset,
        batch_size=BATCH_SIZE,
        generation_config=gen_config,
        return_full_text=False,
    )

    batch_results = []

    # --- INTEGRATED PARSING LOGIC ---
    for task, output in tqdm(zip(tasks, generator), total=len(tasks)):
        raw_response = output[0]['generated_text'].strip()

        # --- Simplified Cleaning ---
        clean_text = re.sub(r'[<>\n\-\*1234567890\.]', ' ', raw_response)

        ignore_words = {
            'and', 'the', 'a', 'an', 'is', 'format', 'word', 'of',
            'associations', 'okay', 'user', 'sure', 'let\'s', 'wants',
            'three', 'provide', 'result'
        }

        parts = [p.strip().lower() for p in clean_text.replace(',', ' ').split() if
                 p.strip().lower() not in ignore_words]

        r1 = parts[0] if len(parts) > 0 else "NaN"
        r2 = parts[1] if len(parts) > 1 else "NaN"
        r3 = parts[2] if len(parts) > 2 else "NaN"

        batch_results.append([task['cue'], task['trial_id'], r1, r2, r3])

        # Write to disk every time we finish a full batch
        if len(batch_results) >= BATCH_SIZE:
            df_write = pd.DataFrame(batch_results)
            df_write.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            batch_results = []

    # Final write for any remaining results in the buffer
    if batch_results:
        df_write = pd.DataFrame(batch_results)
        df_write.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"--- Probing Complete! Results safely stored in {OUTPUT_FILE} ---")


if __name__ == "__main__":
    run_hpc_batch_probe()