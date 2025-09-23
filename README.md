# Model Details

We present MobileLLM-R1, a new series of efficient reasoning models in the MobileLLM family. The release includes two categories of models:

Base models: 
- [MobileLLM-R1-140M-base](https://huggingface.co/facebook/MobileLLM-R1-140M-base/) 
- [MobileLLM-R1-360M-base](https://huggingface.co/facebook/MobileLLM-R1-360M-base/) 
- [MobileLLM-R1-950M-base](https://huggingface.co/facebook/MobileLLM-R1-950M-base/)

Final models: 
- [MobileLLM-R1-140M](https://huggingface.co/facebook/MobileLLM-R1-140M/) 
- [MobileLLM-R1-360M](https://huggingface.co/facebook/MobileLLM-R1-360M/) 
- [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M/)

> **Note**: These models are not general-purpose chat models. They are Supervised Fine-Tuned (SFT) models, specifically trained to address mathematical, programming (Python, C++), and scientific problems.

In addition to the models, we release the complete training recipes and data sources to ensure reproducibility and support further research.

Remarkably, the MobileLLM-R1 950M, pre-trained on only **~2T high-quality tokens** and with fewer than 5T total training tokens, achieves comparable or superior performance to Qwen3 0.6B, which was trained on 36T tokens, across MATH, GSM8K, MMLU, and LiveCodeBench benchmarks. 

Compared to existing fully open-source models, MobileLLM-R1 950M model achieves **~5× higher accuracy on MATH** compared to the Olmo 1.24B model and **~2× higher accuracy** relative to the SmolLM2 1.7B model, despite being substantially smaller in parameter scale. In addition, MobileLLM-R1 950M outperforms both Olmo 1.24B and SmolLM2 1.7B **by a wide margin on coding benchmarks**, establishing a new state-of-the-art among fully open-source models.

# Highlights


### Pretrained Model
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/660f893bae89429c07a32cdb/bZCn1kAhxJUl79cKmz24y.jpeg)

### Token efficiency comparison across pretrained models
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/660f893bae89429c07a32cdb/s1bodFZmznd9vyBj6hi25.jpeg)

### Post-trained Model
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/660f893bae89429c07a32cdb/upEbYrjKnXJc-APYPlN9M.jpeg)



**Model Architecture**:

|  | # Layers | # Attnetion Heads | # KV Heads | Dim | Hidden Dim | Params | 
| --- | --- | --- | --- | --- | --- | --- |
| MobileLLM-R1-140M | 15 | 9 | 3 | 576 | 2048 | 140M | 
| MobileLLM-R1-360M | 15 | 16 | 4 | 1024 | 4096 | 359M | 
| MobileLLM-R1-950M | 22 | 24 | 6 | 1536 | 6144 | 949M | 

|  | Input modalities | Output modalities | Context Length | Vocaburary Size | Shared Embeddings | 
| --- | --- | --- | --- | --- | --- |
| [MobileLLM-R1-140M-base](https://huggingface.co/facebook/MobileLLM-R1-140M-base) | Text | Text | 4k | 128k | Yes | 
| [MobileLLM-R1-360M-base](https://huggingface.co/facebook/MobileLLM-R1-360M-base) | Text | Text | 4k | 128k | Yes | 
| [MobileLLM-R1-950M-base](https://huggingface.co/facebook/MobileLLM-R1-950M-base) | Text | Text | 4k | 128k | Yes | 
| [MobileLLM-R1-140M](https://huggingface.co/facebook/MobileLLM-R1-140M) | Text | Text | 32k | 128k | Yes | 
| [MobileLLM-R1-360M](https://huggingface.co/facebook/MobileLLM-R1-360M) | Text | Text | 32k | 128k | Yes | 
| [MobileLLM-R1-950M](https://huggingface.co/facebook/MobileLLM-R1-950M) | Text | Text | 32k | 128k | Yes | 

# How to use

To load the pretrained model for further finetuning or evaluation:
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-R1-950M")
model = AutoModelForCausalLM.from_pretrained("facebook/MobileLLM-R1-950M")
```

# Inference examples

Transformers

```py
from transformers import pipeline
import torch

model_id = "facebook/MobileLLM-R1-950M"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

# Math problem / default scenario
messages = [
    {
        "role": "system",
        "content": "Please reason step by step, and put your final answer within \\boxed{}."
    },
    {"role": "user", "content": "Compute: $1-2+3-4+5- \\dots +99-100$."},
]

# C++ coding scenario
messages = [
    {
        "role": "system",
        "content": (
            "\nYou are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.\n\n"
            "Please use c++ programming language only.\n"
            "You must use ```cpp for just the final solution code block with the following format:\n"
            "```cpp\n# Your code here\n```\n"
        )
    },
    {"role": "user", "content": "Write a C++ program that prints 'Hello, World!'."},
]

# Python coding scenario
messages = [
    {
        "role": "system",
        "content": (
            "\nYou are a helpful and harmless assistant. You should think step-by-step before responding to the instruction below.\n\n"
            "Please use python programming language only.\n"
            "You must use ```python for just the final solution code block with the following format:\n"
            "```python\n# Your code here\n```\n"
        )
    },
    {"role": "user", "content": "Write a Python function that returns the square of a number."},
]

outputs = pipe(
    messages,
    max_new_tokens=8192,
)
print(outputs[0]["generated_text"][-1])
```

You can also run inference with vLLM. You only need to register the model architecture Llama4ForCausalLM with the vLLM ModelRegistry.
```bash
from vllm.model_executor.models.llama4 import Llama4ForCausalLM
from vllm.model_executor.models.registry import ModelRegistry
ModelRegistry.register_model("Llama4ForCausalLM", Llama4ForCausalLM)
```


# Evaluation

## MobileLLM-R1 base model
| Model | Size | MATH500 | GSM8K | MBPP | HumanEval | CommonSense Avg. | MMLU |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | 4-shot <br> em | 8-shot <br> em | 3-shot <br> pass@1 | 0-shot <br> pass@1 | 0-shot <br> accuracy | 5-shot <br> accuracy |
|  | 
| *<150M* |  |  |  |  |  |  |  | 
| SmolLM2-135M-base | 135M | 0.4 | 1.8 | 3.8 | 0.0 | **50.7** | -- |
| **MobileLLM-R1-140M-base** | 140M | **4.6** | **16.3** | **5.4** | **15.9** | 44.3 | -- |
|  | 
| *150M - 400M* |  |  |  |  |  |  |  | 
| Gemma-3-270M-pt | 268M | 0.6 | 1.1 | 2.0 | 3.1 | 48.4 | 26.5 |
| SmolLM2-360M-base | 362M | 1.8 | 5.0 | **19.4** | 0.0 | **56.6** | 24.7 |
| **MobileLLM-R1-360M-base** | 359M | **13.4** | **39.4** | **20.8** | **32.9** | 51.0 | **26.8** |
|  | 
| *400M - 1B* |  |  |  |  |  |  |  | 
| Qwen2.5-0.5B-base | 494M | 14.8 | 41.8 | 29.6 | 28.1 | 52.3 | 47.5 |
| Qwen3-0.6B-base | 596M | **29.8** | 60.9 | **39.0** | 30.5 | 55.3 | **52.4** |
| **MobileLLM-R1-950M-base** | 949M | 26.8 | **61.6** | **39.2** | **46.3** | **58.6** | 47.4 |
|  | 
| *> 1B* |  |  |  |  |  |  |  | 
| Gemma-3-1B-pt | 1.0B | 0.6 | 2.4 | 9.4 | 6.1 | 57.3 | 26.1 |
| LLaMA3.2-1B-base | 1.24B | 1.6 | 6.8 | 26.6 | 17.1 | 58.4 | 32.0 |
| OLMo-2-0425-1B-base | 1.48B | 5.2 | 39.8 | 7.8 | 6.7 | 61.0 | 42.4 |
| Qwen2.5-1.5B-base | 1.54B | 31.0 | 68.4 | 44.6 | 36.6 | 58.7 | 61.2 |
| SmolLM2-1.7B-base | 1.71B | 11.6 | 31.8 | 35.4 | 0.6 | 62.9 | 50.0 |
| Qwen3-1.7B-base | 2.03B | 38.5 | 76.2 | 56.4 | 47.6 | 60.9 | 62.1 |


Here, CommonSense Avg. denotes an average of 8 tasks in CommonSense Reasoning benchmarks including ARC-easy, ARC-challenge, BoolQ, PIQA, SIQA, HellaSwag, OBQA, and WinoGrand. Models with fewer than 150M parameters do not yield reliable MMLU scores and are therefore denoted as '—'. 
 
## MobileLLM-R1 post-trained model

 | Model | Size | MATH500 | GSM8K | AIME'24 | AIME'25 | LiveCodeBench-v6 | 
 | --- | --- | --- | --- | --- | --- | --- |
 |  |  | 0-shot <br> pass@1 | 0-shot <br> pass@1 | 0-shot <br> pass@1, n=64 | 0-shot <br> pass@1, n=64 | 0-shot <br> pass@1, n=16 |
 |  |
 | *<150M* |  |  |  |  |  |  | 
 | SmolLM2-135M-Instruct | 135M | 3.0 | 2.4 | -- | -- | 0.0 | 
 | **MobileLLM-R1-140M** | 140M | **6.2** | **4.1** | -- | -- | **1.7** | 
 |  |
 | *150M - 400M* |  |  |  |  |  |  | 
 | Gemma-3-270m-it | 268M | 6.8 | 8.4 | -- | -- | 0.0 | 
 | SmolLM2-360M-Instruct | 362M | 3.4 | 8.1 | -- | -- | 0.7 | 
 | **MobileLLM-R1-360M** | 359M | **28.4** | **24.5** | -- | -- | **5.1** | 
 |  |
 | *400M - 1B* |  |  |  |  |  |  | 
 | Qwen2.5-0.5B-Instruct | 494M | 31.2 | 48.1 | 0.1 | 0.3 | 3.6 | 
 | Qwen3-0.6B | 596M | 73.0 | **79.2** | 11.3 | **17.0** | 14.9 | 
 | **MobileLLM-R1-950M** | 949M | **74.0** | 67.5 | **15.5** | 16.3 | **19.9** | 
 |  |
 | *> 1B* |  |  |  |  |  |  | 
 | Gemma-3-1B-it | 1.0B | 45.4 | 62.9 | 0.9 | 0.0 | 2.0 | 
 | LLaMA3.2-1B-Instruct | 1.24B | 24.8 | 38.8 | 1.1 | 0.2 | 4.1 | 
 | OLMo-2-0425-1B-Instruct | 1.48B | 19.2 | 69.7 | 0.6 | 0.1 | 0.0 | 
 | OpenReasoning-Nemotron-1.5B | 1.54B | 83.4 | 76.7 | 49.7 | 40.4 | 28.3 | 
 | DeepSeek-R1-Distill-Qwen-1.5B | 1.54B | 83.2 | 77.3 | 29.1 | 23.4 | 19.9 | 
 | Qwen2.5-1.5B-Instruct | 1.54B | 54.0 | 70.0 | 2.5 | 0.9 | 7.9 | 
 | SmolLM2-1.7B-Instruct | 1.71B | 19.2 | 41.8 | 0.3 | 0.1 | 4.4 | 
 | Qwen3-1.7B | 2.03B | 89.4 | 90.3 | 47.0 | 37.0 | 29.8 | 

For AIME, we evaluate models across 64 runs and report the average accuracy. For LiveCodeBench, results are reported as the average accuracy across 16 runs. Models with fewer than 400M parameters do not produce reliable AIME scores and are therefore denoted as '—'.


# Training

## Training Process
![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/660f893bae89429c07a32cdb/ThVFzsaaGa4gQ3iha5CKM.jpeg)

### Training stages and hyperparameter details

In the pretraining phase, MobileLLM-R1 models are randomly initialized and optimized using the Adam optimizer with hyperparameters (β_1, β_2, ε) = (0.9, 0.95, 1e-8), coupled with a weight decay coefficient of 0.1. The learning rate follows a 2k-step warmup schedule and then decays linearly from its peak to 10\% of the maximum. 

In the mid-training phase, we use Adam optimizer with learning rate linearly decays from its maximum value to zero. We employ knowledge distillation with Llama-3.1-8B-Instruct model as the teacher, where the student is trained via minimizing the KL divergence between its output logits and the teacher logits.

In the post-training phase, we use the Adam optimizer with zero weight decay. The learning rate warmup ratio is set to 0.03 for general-purpose SFT and 0.1 for reasoning-specific SFT, and it linearly decays from its maximum value to zero. Full training hyperparameters are provided in the table below.

| Stage | Phase | Tokens / Samples | BS | Sequence Length | Steps | LR | #GPUs | Training Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pre-training | Phase1 | 2T tokens | 16 | 2k | 500k | 4.00E-03 | 16 x 8 | 4-5 days |
|  | Phase2 | 2T tokens  | 16 | 2k | 500k | 4.00E-03 | 16 x 8 | 4-5 days |
| Mid-training | Phase1 | 100B tokens  | 4 | 4k | 50K | 3.60E-04 | 16 x 8 | 1-2 days |
|  | Phase2 | 100B tokens | 4 | 4k | 50K | 3.60E-04 | 16 x 8 | 1-2 days |
| Post-training | General SFT | 866K samples | 4 | 4k | 2 epochs | 5.00E-06 | 16 x 8 | ~2h |
|  | Reasoning SFT | 6.2M samples | 8 | 32k | 4 epochs | 8.00E-05 | 16 x 8 | ~2.5days |

<!-- ## Data Mix

### Pre-training

| Dataset | Rows | Tokens (B) | Phase1 Mix Ratio | Phase2 Mix Ratio |
| --- | --- | --- | --- | --- |
| [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) | 206,640,114 | 263.8 | 10.66% | 0.52% |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 6,117,786 | 12.6 | 6.93% | 23.33% |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1,279,107,432 | 1300 | 63.75% | 54.83% |
| [Wiki](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/wiki) | 7,222,303 | 3.7 | 5.03% | 0.14% |
| [Arxiv](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/blob/main/urls/arxiv.txt) | 1,533,917 | 28 | 6.36% | 1.32% |
| [StackExchange](https://data.together.xyz/redpajama-data-1T/v1.0.0/stackexchange/stackexchange.jsonl) | 29,249,120 | 19.6 | 5.03% | 0.86% |
| [Algebraic stack](https://huggingface.co/datasets/EleutherAI/proof-pile-2/tree/main/algebraic-stack) | 3,404,331 | 12.6 | 2.25% | 1.26% |
| [Nemotron science](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/blob/main/SFT/science/science.jsonl) | 708,920 | 2 | -- | 0.03% |
| [Nemotron code](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/blob/main/SFT/code/code_v1.1.jsonl) | 10,108,883 | 16 | -- | 0.72% |
| [Nemotron math](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/blob/main/SFT/math/math_v1.1.jsonl) | 22,066,397 | 15 | -- | 3.01% |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | 31,064,744 | 25 | -- | 2.70% |
| [Facebook natural reasoning](https://huggingface.co/datasets/facebook/natural_reasoning) | 1,145,824 | 1.8 | -- | 3.18% |
| [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/finemath-3plus) | 48,283,984 | 34 | -- | 8.01% |
| [peS2o](https://huggingface.co/datasets/allenai/peS2o) | 38,800,000 | 50 | -- | 0.08% |
| **Total** |  |  | 100% | 100% |




### Mid-training


 | Dataset | Subset | Rows (M) | Phase1 Mix Ratio | Phase2 Mix Ratio | 
 | --- | --- | --- | --- | --- |
 | [Dolmino](https://huggingface.co/datasets/allenai/dolmino-mix-1124) | DCLM Baseline | 606 | 37.03% | 6.51% | 
 |  | FLAN | 57.3 | 4.10% | 0.72% | 
 |  | peS2o | 38.8 | 11.41% | 2.01% | 
 |  | Wiki | 6.17 | 2.66% | 0.47% | 
 |  | StackExchange | 2.48 | 2.12% | 2.00% | 
 |  | Math | 21 | 11.63% | 29.10% | 
 | Nemotron | [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) | 882 | 20.69% | 29.10% | 
 |  | [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) | 144 | 3.45% | 19.40% | 
 | StarCoder | [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) | 206 | 6.90% | 9.70% | 
 | Benchmark training set | [TriviaQA (train)](https://huggingface.co/datasets/mandarjoshi/trivia_qa/tree/main/rc) <br> [OBQA (train)](https://huggingface.co/datasets/allenai/openbookqa/blob/main/main/train-00000-of-00001.parquet) <br> [NaturalQuestions (train)](https://github.com/google-research-datasets/natural-questions/blob/master/nq_open/NQ-open.train.jsonl) <br> [PIQA (train)](https://github.com/ybisk/ybisk.github.io/blob/master/piqa/data/train.jsonl) <br> [GSM8K (train)](https://huggingface.co/datasets/openai/gsm8k/blob/main/main/train-00000-of-00001.parquet) <br> [BoolQ (train)](https://huggingface.co/datasets/google/boolq/blob/main/data/train-00000-of-00001.parquet) <br> [ARC-Easy (train)](https://huggingface.co/datasets/allenai/ai2_arc/blob/main/ARC-Easy/train-00000-of-00001.parquet) <br> [ARC-Challenge (train)](https://huggingface.co/datasets/allenai/ai2_arc/blob/main/ARC-Challenge/train-00000-of-00001.parquet) | ~0.01 | 0 | 0.97% | 
 | Total |  |  | 100.00% | 100.00% | 
 
### Post-training 
 | Phase | Dataset | Rows | 
 | --- | --- | --- |
 | General SFT | [Tulu-3-sft-olmo-2-mixture-0225](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225) | 866K samples |
 | Reasoning SFT | [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | 3.2M samples |
 | | [OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2) | 803K samples |
 | | [OpenCodeReasoning-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | 2.16M samples | -->

# Citation

If you find our model useful for your research, please consider citing:

    @misc{mobilellm_r1_2025,
      title={MobileLLM-R1: Model Card},
      author={Changsheng Zhao*, Ernie Chang*, Zechun Liu*, Chia-Jung Chang, Wei Wen, Chen Lai, Rick Cao, Yuandong Tian, Raghuraman Krishnamoorthi, Yangyang Shi, Vikas Chandra},
      year={2025},
      url = {https://huggingface.co/mobilellm-r1}
    }

# Contact
Changsheng Zhao, Meta Inc (cszhao at meta dot com)

Ernie Chang, Meta Inc (erniecyc at meta dot com)

Zechun Liu, Meta Inc (zechunliu at meta dot com)

# License

MobileLLM-R1 is FAIR NC licensed as of now