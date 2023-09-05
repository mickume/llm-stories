min_length = 100

# Dataset
context_length = 2048
batch_size = 1000

# Model
model_name = 'bigscience/bloom-3b'

lora_r = 16 # attention heads
lora_alpha = 32 # alpha scaling
lora_dropout = 0.05
lora_bias = "none"
lora_task_type = "CAUSAL_LM" # set this for CLM or Seq2Seq

## Trainer config
per_device_train_batch_size = 2
gradient_accumulation_steps = 2
warmup_steps = 100
num_train_epochs=3
weight_decay=0.1
learning_rate = 2e-4
fp16 = True
logging_steps = 100
overwrite_output_dir = True
evaluation_strategy = "no"
save_strategy = "steps"
push_to_hub = True
hub_strategy = "checkpoint"

## Data collator
mlm =False

# Other
output_dir = "./cache"