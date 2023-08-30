from typing import Mapping, Any

from datasets import Dataset, load_dataset

from torch import float32, nn, exp, cuda
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model

from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling


# training utils
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(float32)


def prepare_model(model):
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(float32)
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


def compute_perplexity(pred):
    pass
    # loss = pred.losses.mean()
    # perplexity = exp(loss)
    # return {"perplexity": perplexity}


# Model


class LLMStories():

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    def train(self, hf_repo: str, lora_config: Mapping[str, Any], trainer_config: Mapping[str, Any], mlm: bool) -> None:

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", load_in_8bit=True)
        model = prepare_model(model)
        model = get_peft_model(model, LoraConfig(**lora_config))
        print(
            f"Model trainable parameters:\n {print_trainable_parameters(model)}")

        dataset = load_dataset(f"{hf_repo}_tk")
        print(f"Train dataset downloaded:\n {dataset['train']}")
        print(
            f"Number of tokens for the training: {dataset['train'].num_rows*len(dataset['train']['input_ids'][0])}")

        trainer = Trainer(
            model=model,
            train_dataset=dataset['train'],
            args=TrainingArguments(**trainer_config),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        )
        model.config.use_cache = False  # silence warnings

        trainer.train()

        model.config.use_cache = True
        model.push_to_hub(repo_id=f"{hf_repo}_model")
        tokenizer.push_to_hub(repo_id=f"{hf_repo}_model")

    def evaluate():
        pass

    def generate(self, prompt: str, hf_repo: str, max_new_tokens: int, temperature: float, do_sample: bool) -> None:
        # Import the model
        config = PeftConfig.from_pretrained(f"{hf_repo}_model")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path)

        # Load the Lora model
        self.model = PeftModel.from_pretrained(model, f"{hf_repo}_model")

        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print(tokenizer.decode(tokens[0]))
