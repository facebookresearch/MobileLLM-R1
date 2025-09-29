# Post-Training

This directory contains the minimal code required to reproduce the post-training stages of MobileLLM-R1.

## Installation
Install [TRL library](https://github.com/huggingface/trl/tree/main) and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
```
pip install trl
pip install liger-kernel
```
## Usage
To run the general SFT stage, first add chat_template([example](https://huggingface.co/facebook/MobileLLM-R1-950M/blob/main/chat_template.jinja)) to base model. And then use the following command:
```
sh run_general_sft.sh
```
To run the reasoning SFT stage, first increase the model's max_position_embeddings to 32,768 by scaling ```rope_theta``` by 8x. Then run:
```
sh run_reasoning_sft.sh
```
