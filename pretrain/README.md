# Pre-Training/Mid-Training

This directory contains the essential code needed to reproduce the pre-training and mid-training phases of MobileLLM-R1.

## Installation
Install pytorch and transformers>=4.55.0.
```
pip install -r requirement.txt
```
## Usage
### Step 1. Data preparation
Organize the pretrain dataset folder into shards named 1, 2, 3, 4, ..., up to #nodes, where #nodes is the total number of nodes used for training. Each node should have its own data shard. Inside each shard folder, create subfolders named after the data mix and its sample weight. For example:
```
basepath/
├── 1/
│   ├── dclm:0.537/
│   │   └── data.jsonl
│   ├── flan:0.0094/
│   │   └── data.jsonl
├── 2/
│   ├── dclm:0.537/
│   │   └── data.jsonl
│   ├── flan:0.0094/
│   │   └── data.jsonl
...
├── #nodes/
│   ├── dclm:0.537/
│   │   └── data.jsonl
│   ├── flan:0.0094/
│   │   └── data.jsonl
```
Each line in a jsonl file should be a JSON object containing the key "text".

### Step 2. Training
The script run_pretrain.sh is provided to initiate training on a 1x8 node setup using torchrun. This script can be modified to adjust the --nnodes parameter and other settings to suit different multi-node configurations, such as those using slurm or torchx. The learning rate in the script is for 1x8 node with a batch size of 32. If you increase the number of nodes or the batch size, you need to increase the learning rate linearly.

**Step 2. Training**

1. In the `pretrain.sh` file, specify the following arguments:
   - `--train_data_local_path`: Path to the organized data directory from Step 1.
   - `--input_model_filename`: Path to the model config, e.g., `./configs/{model_size}/`.

2. Start training by running:
   ```
   bash run_pretrain.sh
   ```
