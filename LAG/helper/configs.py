model_options = {
        "nemotron" : [
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
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
            "pipeline_parallel_size" :2,
            },
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8" : {
            "enable_prefix_caching" : True,
            "pipeline_parallel_size" :2,
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
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8" : {
            "temperature" : 1.0,
            "top_p" : 0.95,
            },
        }

model_default= {
                "enable_prefix_caching" : True,
                "tensor_parallel_size" : 1,
                }

sampler_default= {
        "temperature" : 1.0,
        }
