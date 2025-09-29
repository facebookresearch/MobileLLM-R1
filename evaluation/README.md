# Evaluation

This directory contains code for reproducing math-related metrics in the zero-shot evaluation of the MobileLLM-R1 final model. The codebase is adapted from this [repository](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation).

## Installation
First install vLLM
```
pip install vllm
```

Next, install the dependencies.
```
pip install -r requirements.txt
```
## Usage
To run the math evaluation, first download the evaluation dataset from [here](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation/data)[^1] and place it in the `data` folder. Then, run the following command:
```
sh run_eval.sh
```

[^1]: We suggest utilizing the [OpenAI test subset](https://github.com/openai/prm800k) for evaluating MATH performance, since the original `MATH` test set has already been included in public training sets such as PRM800k.

## Acknowledgement

- https://github.com/microsoft/ToRA
- https://github.com/openai/prm800k
- https://github.com/wellecks/lm-evaluation-harness
- https://github.com/deepseek-ai/DeepSeek-Math
- https://github.com/QwenLM/Qwen2.5-Math
- https://github.com/ZubinGou/math-evaluation-harness