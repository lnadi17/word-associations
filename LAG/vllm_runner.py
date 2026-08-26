import argparse
import logging
from datetime import datetime

import questionary 
import pandas as pd

from vllm import LLM, SamplingParams

from helper.prompting import prepare_conversation
from helper.cues import generate_dataloader
import helper.configs as configs

MAX_TOKENS = 256
SAMPLES_PER_CUE = 100

def get_model_name(model):
    return model.split("/")[-1]


def setup_logger(model):
    model = get_model_name(model)
    START_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename=f"./logs/{model}_{START_TIME}.log",
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def main(dataloader, model, model_config, sampler_config):
    llm = LLM(model=model, **model_config)

    sampling_params = SamplingParams(
        n=SAMPLES_PER_CUE,
        max_tokens=MAX_TOKENS,
        **sampler_config
    )

    tokenizer = llm.get_tokenizer()
    model_name = get_model_name(model)
    for (batch_idx, batch) in enumerate(dataloader):
        conversations = prepare_conversation(batch)

        prompts = [ tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False) for conversation in conversations ]

        outputs = llm.generate(prompts, sampling_params)
        rows = []
        for prompt_idx, req_output in enumerate(outputs):
            for var_idx, completion in enumerate(req_output.outputs):
                rows.append({
                    "prompt_index": prompt_idx,
                    "variation_index": var_idx,
                    "cue" : batch[prompt_idx],
                    "prompt_text": req_output.prompt,
                    "response": completion.text,
                    "finish_reason": completion.finish_reason,
                    })
        df = pd.DataFrame(rows)
        df.to_csv(f"./outputs/results/{model_name}_{batch_idx}.csv", index=False)

    return 0

if "__main__" == __name__:
    parser = argparse.ArgumentParser("vLLM runner")
    parser.add_argument("--cues", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--hf_repo", type=str, default="")

    args = parser.parse_args()

    if(args.hf_repo != ""):
        model = args.hf_repo
    else:
        if (args.model_type == ""):
            model_type = questionary.select(message="select model type",
                                            choices= list(configs.model_options.keys())).ask()
        else:
            model_type = args.model_type

        model = questionary.select(message="select particular model",
                                   choices=configs.model_options[model_type]).ask()

    if model in configs.models.keys():
        model_config = configs.models[model]
        sampler_config = configs.samplers[model]
    else:
        print("USING DEFAULT CONFIG: USE ONLY FOR DEBUG!")
        model_config = configs.model_default
        sampler_config = configs.sampler_default
    dataloader = generate_dataloader(args.cues)

    setup_logger(model)
    start_time = datetime.now()
    outputs = main(dataloader, model, model_config, sampler_config)
    elapsed_time = datetime.now() - start_time
    logging.info(f"\n====\nELAPSED TIME: {elapsed_time}")


    

