import os
import logging
import argparse

from . import config
from trainer.model import LLMStories

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train an LLM with Lora.")
    # required
    parser.add_argument("--model-name", type=str, default=config.model_name, help="The name of the model to train.")
    parser.add_argument("--repo", type=str, help="The name of the HuggingFace repo to push the model to.")
    parser.add_argument("--token", type=str)
    # optional
    parser.add_argument("--mlm", type=bool, default=config.mlm, help="Whether to use MLM or not for the training.")
    parser.add_argument('--lora-r', type=float, default=config.lora_r, help="The Lora parameter r, the number of heads.")
    parser.add_argument('--lora-alpha', type=float, default=config.lora_alpha, help="Lora parameter.")
    parser.add_argument('--lora-dropout', type=float, default=config.lora_dropout, help="Lora dropout.")
    parser.add_argument('--lora-bias', type=str, default=config.lora_bias, help="Lora bias.")
    parser.add_argument('--lora-task-type', type=str, default=config.lora_task_type, help="Lora task type.")
    parser.add_argument('--per-device-train-batch-size', type=int, default=config.per_device_train_batch_size, help="The batch size per device for the training.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=config.gradient_accumulation_steps, help="The number of gradient accumulation steps.")
    parser.add_argument('--warmup-steps', type=int, default=config.warmup_steps, help="The number of warmup steps.")
    parser.add_argument('--weight-decay', type=float, default=config.weight_decay, help="The weight decay.")
    parser.add_argument('--num-train-epochs', type=float, default=config.num_train_epochs, help="The number of training epochs.")
    parser.add_argument('--learning-rate', type=float, default=config.learning_rate, help="The learning rate.")
    parser.add_argument('--fp16', type=bool, default=config.fp16, help="Whether to use fp16 or not.")
    parser.add_argument('--logging-steps', type=int, default=config.logging_steps, help="The number of logging steps.")
    parser.add_argument('--output-dir', type=str, default=config.output_dir, help="The output directory.")
    parser.add_argument('--overwrite-output_dir', type=bool, default=config.overwrite_output_dir, help="Whether to overwrite the output directory.")
    parser.add_argument('--save-strategy', type=str, default=config.save_strategy, help="The saving strategy.")
    parser.add_argument('--evaluation-strategy', type=str, default=config.evaluation_strategy, help="The evaluation strategy.")
    parser.add_argument('--push-to-hub', type=bool, default=config.push_to_hub, help="Whether to push the model to the HuggingFace Hub.")
    args = parser.parse_args()

    os.environ['HUGGING_FACE_HUB_TOKEN'] = args.token

    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, 
        'bias': args.lora_bias,
        "task_type": args.lora_task_type,
    }

    trainer_config = {
        "per_device_train_batch_size": args.per_device_train_batch_size, 
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate, 
        "fp16": args.fp16,
        "logging_steps": args.logging_steps, 
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "push_to_hub": args.push_to_hub
    }

    model = LLMStories(args.model_name)
    model.train(
        hf_repo=args.repo,
        lora_config=lora_config,
        trainer_config=trainer_config,
        mlm=args.mlm
    )