model_options = {
        "nemotron-3.5" : [
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            ],
        "nemotron-3" : [
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            ],
        "gemma-4" : [
            "google/gemma-4-E2B-it",
            "google/gemma-4-E4B-it",
            "google/gemma-4-12B-it",
            "google/gemma-4-26B-A4B-it",
            "google/gemma-4-31B-it",
            ],
        "qwen" : [
            "Qwen/Qwen3.8-27B",
            "Qwen/Qwen3.8-27B-FP8"
            ]
        }


models= {
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" : {
            "enable_prefix_caching" : True,
            "pipeline_parallel_size" : 2,
            },
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "google/gemma-4-31B-it" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "google/gemma-4-26B-A4B-it" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "google/gemma-4-12B-it" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "google/gemma-4-E4B-it" : {
            "enable_prefix_caching" : True,
            "pipeline_parallel_size" : 2,
            },
        "google/gemma-4-E2B-it" : {
            "enable_prefix_caching" : True,
            "pipeline_parallel_size" : 2,
            },
        "Qwen/Qwen3.8-27B" : {
            "enable_prefix_caching" : True,
            "tensor_parallel_size" : 2,
            },
        "Qwen/Qwen3.8-27B-FP8" : {
            "enable_prefix_caching" : True,
            "pipeline_parallel_size" : 2,
            },
        }

samplers= {
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            },
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" : {
            "temperature" : 1.0,
            "top_p" : 1.0,
            },
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            },
        "google/gemma-4-31B-it" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 64,
            },
        "google/gemma-4-26B-A4B-it" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 64,
            },
        "google/gemma-4-12B-it" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 64,
            },
        "google/gemma-4-E4B-it" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 64,
            },
        "google/gemma-4-E2B-it" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 64,
            },
        "Qwen/Qwen3.8-27B" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 20,
            "min_p" : 0.0,
            "presence_penalty" : 0.0,
            "repetition_penalty" : 1.0,
            },
        "Qwen/Qwen3.8-27B-FP8" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            # "top_k" : 20,
            "min_p" : 0.0,
            "presence_penalty" : 0.0,
            "repetition_penalty" : 1.0,
            },
        }

model_default= {
                "enable_prefix_caching" : True,
                "tensor_parallel_size" : 1,
                }

sampler_default= {
        "temperature" : 1.0,
        }
