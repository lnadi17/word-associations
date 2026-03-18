import pandas as pd
import torch
from transformers import AutoTokenizer, pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import json
import re

# --- Configuration ---
MODEL_PATH = "tbilisi-ai-lab/kona2-12B-Instruct"
INPUT_FILE = "master_cues.json"
OUTPUT_FILE = "qwen_results.csv"
NUM_SAMPLES = 100 #Samples per Cue
BATCH_SIZE = 64
TOKENS = 256
TEMP = 1 #Temperature

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


    pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16
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
        # Create a set of tuples: e.g., {("apple", 1), ("apple", 2)...}
        completed_tasks = set(zip(df_done['cue'].astype(str), df_done['trial_id'].astype(int)))
        print(f"Found {len(completed_tasks)} completed trials in existing CSV.")
    else:
        # Create the file with headers
        pd.DataFrame(columns=['cue', 'trial_id', 'response_1', 'response_2', 'response_3']).to_csv(OUTPUT_FILE,
                                                                                                   index=False)

    # --- 3. Build the Task Matrix ---
    print("Building task queue...")
    tasks = []

    # The strict prompt you provided
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
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        for trial in range(1, NUM_SAMPLES + 1):
            if (cue, trial) not in completed_tasks:
                tasks.append({
                    'cue': cue,
                    'trial_id': trial,
                    'prompt': prompt
                })

    if not tasks:
        print("All 100 trials for all cues are complete! Exiting.")
        return

    print(f"Queue built: {len(tasks)} inferences remaining.")

    # --- 4. The High-Speed Batch Inference Loop ---
    dataset = PromptDataset(tasks)

    print(f"--- Starting Batch Probing (Batch Size: {BATCH_SIZE}) ---")

    generator = pipe(
        dataset,
        batch_size=BATCH_SIZE,
        max_new_tokens=TOKENS,
        do_sample=True,
        temperature=TEMP,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False
    )

    batch_results = []

    for task, output in tqdm(zip(tasks, generator), total=len(tasks)):
        raw_response = output[0]['generated_text'].strip()

        # --- Cleaning ---
        if "</think>" in raw_response:
            clean_response = raw_response.split("</think>")[-1].strip()
        else:
            clean_response = raw_response.replace("<think>", "").strip()

        clean_text = re.sub(r'[<>\n\-\*1234567890\.]', ' ', clean_response)

        ignore_words = {
            'and', 'the', 'a', 'an', 'is', 'format', 'word', 'of',
            'associations', 'okay', 'user', 'sure', 'let\'s', 'wants',
            'three', 'provide', 'think', 'see', 'stimulus', 'result'
        }

        parts = [p.strip().lower() for p in clean_text.replace(',', ' ').split() if
                 p.strip().lower() not in ignore_words]

        r1 = parts[0] if len(parts) > 0 else "NaN"
        r2 = parts[1] if len(parts) > 1 else "NaN"
        r3 = parts[2] if len(parts) > 2 else "NaN"


        batch_results.append([task['cue'], task['trial_id'], r1, r2, r3])

        # Write to disk every time we finish a full batch to save I/O overhead
        if len(batch_results) >= BATCH_SIZE:
            df_write = pd.DataFrame(batch_results)
            df_write.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            batch_results = []  # Clear the buffer

    # Write any remaining results in the buffer
    if batch_results:
        df_write = pd.DataFrame(batch_results)
        df_write.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print(f"--- Probing Complete! Results safely stored in {OUTPUT_FILE} ---")


if __name__ == "__main__":
    run_hpc_batch_probe()
