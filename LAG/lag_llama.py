import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json
import re

# --- Configuration ---
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
INPUT_FILE = "master_cues.json"
OUTPUT_FILE = "llama_results_3.1-8B.csv"
NUM_SAMPLES = 100
BATCH_SIZE = 64
TOKENS = 256
TEMP = 1.0


class PromptDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]['prompt']


def run_hpc_batch_probe():
    print(f"--- Loading Tokenizer and {MODEL_PATH} ---")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        padding_side="left"
    )
    # Llama 3 requires a padding token definition
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return

    with open(INPUT_FILE, 'r') as f:
        all_cues = json.load(f)

    # Resume Logic
    completed_tasks = set()
    if os.path.exists(OUTPUT_FILE):
        df_done = pd.read_csv(OUTPUT_FILE, usecols=['cue', 'trial_id'])
        completed_tasks = set(zip(df_done['cue'].astype(str), df_done['trial_id'].astype(int)))
        print(f"Found {len(completed_tasks)} completed trials.")
    else:
        pd.DataFrame(columns=['cue', 'trial_id', 'r1', 'r2', 'r3']).to_csv(OUTPUT_FILE, index=False)

    print("Building task queue...")
    tasks = []

    # YOUR EXACT RESEARCH PROMPT
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
        # Using chat template is crucial for Llama 3 instruction following
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        for trial in range(1, NUM_SAMPLES + 1):
            if (str(cue), trial) not in completed_tasks:
                tasks.append({'cue': cue, 'trial_id': trial, 'prompt': prompt})

    if not tasks:
        print("All trials complete!");
        return

    dataset = PromptDataset(tasks)
    generator = pipe(
        dataset,
        batch_size=BATCH_SIZE,
        max_new_tokens=TOKENS,
        do_sample=True,
        temperature=TEMP,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )

    print(f"--- Starting Llama Probing ({len(tasks)} inferences) ---")
    batch_results = []

    ignore_words = {
        'and', 'the', 'a', 'an', 'is', 'format', 'word', 'of',
        'associations', 'okay', 'user', 'sure', 'let\'s', 'wants',
        'three', 'provide', 'think', 'see', 'stimulus', 'result'
    }

    for task, output in tqdm(zip(tasks, generator), total=len(tasks)):
        raw_response = output[0]['generated_text'].strip()

        # Cleaner regex: keeps letters and commas, strips everything else
        clean_text = re.sub(r'[^a-zA-Z, ]', ' ', raw_response)

        # Split by comma first, then handle fallback
        if ',' in clean_text:
            parts = [p.strip().lower() for p in clean_text.split(',') if p.strip()]
        else:
            parts = [p.strip().lower() for p in clean_text.split() if p.strip()]

        # Final filter against ignore_words and sentence-like blobs
        final_parts = [p for p in parts if p not in ignore_words and len(p.split()) <= 2]

        r1 = final_parts[0] if len(final_parts) > 0 else "NaN"
        r2 = final_parts[1] if len(final_parts) > 1 else "NaN"
        r3 = final_parts[2] if len(final_parts) > 2 else "NaN"

        batch_results.append([task['cue'], task['trial_id'], r1, r2, r3])

        if len(batch_results) >= BATCH_SIZE:
            pd.DataFrame(batch_results).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            batch_results = []

    if batch_results:
        pd.DataFrame(batch_results).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"--- Done! Results in {OUTPUT_FILE} ---")


if __name__ == "__main__":
    run_hpc_batch_probe()