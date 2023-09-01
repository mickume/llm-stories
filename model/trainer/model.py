from typing import Mapping, Any

from datasets import Dataset, load_dataset

from torch import float32, nn, exp, cuda
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model

from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.models.nat.modeling_nat import NatPatchEmbeddings

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

class LLMStoriesTrainer():

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


class LLMStoryTeller():

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        # Load the finetuned model and associated tokenizer
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path)

        # Load the Lora model
        self.model = PeftModel.from_pretrained(model, model_name)

    def create_sequence(self, prompt: str, temp: float, max_sequence: int, top_k: int, top_p: float):
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')

        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_sequence,
            temperature=temp,
            do_sample=True,
            top_k=top_k,
            top_p=top_p
        )
        return tokens

    def generate(self, prompt: str, max_tokens: int, overlap: int = 6, temp: float = 0.6, max_sequence: int = 25, top_k: int = 0, top_p: float = 0.92, debug: bool = False) -> str:

        n = 0
        seq = []

        # first seq is a special case, it includes the initial prompt ...
        output = self.create_sequence(prompt, temp, max_sequence, top_k, top_p)
        seq.append(output[0])
        tokens = len(output[0])

        if debug:
            print(
                f"{n}: {self.tokenizer.decode(output[0], skip_special_tokens=True)}")

        while tokens < max_tokens:
            n += 1
            np = self.tokenizer.decode(
                output[0][(-1*overlap):], skip_special_tokens=True)

            output = self.create_sequence(np, temp, max_sequence, top_k, top_p)
            seq.append(output[0][:(-1*overlap)])
            tokens += len(seq[n])

            if debug:
                print(
                    f"{n}: {self.tokenizer.decode(seq[n], skip_special_tokens=True)}")

        para = []
        for s in seq:
            para += s

        return self.tokenizer.decode(para, skip_special_tokens=True)
