import torch
import transformers

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from huggingface_hub import login, HfApi
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}

    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result
    
def main():
    login(token="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")

    # load dataset
    dataset = load_dataset("shivank21/llm_efficiency", split="train").train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
#    print(train_dataset)
#    print(eval_dataset)

    # load model
    base_model_id = "ArianAskari/NeuralHermes-2.5-Mistral-7B"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
    model.config.window = 2048

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_eos_token=True)

    tokenizer.pad_token = tokenizer.eos_token

    def get_dataset(dataset):
        prompt = ("""{prompt}""")

        def apply_prompt_template(sample):
            return {
                "text": prompt.format(
                    prompt=sample["prompt"]
                )
            }

        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

        dataset = dataset.map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features),
        ).map(Concatenator(), batched=True)
        return dataset

    tokenized_train_dataset = get_dataset(train_dataset)
    tokenized_val_dataset = get_dataset(eval_dataset)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.10,  
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    project = "mistral-rlhf-finetune-shivank"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    tokenizer.pad_token = tokenizer.eos_token

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=10,
            gradient_accumulation_steps=4,
            max_steps=300,
            learning_rate=2e-5, 
            logging_steps=20,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_dir="./logs",        
            save_strategy="steps",       
            save_steps=20,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=20,               
            do_eval=True,                # Perform evaluation at the end of training
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  
    trainer.train()

    api = HfApi()
    api.upload_folder(
        folder_path="mistral-mistral-finetune/checkpoint-300",
        repo_id="shivank21/mistral-rlhf-7b-tuned",
        repo_type='model',
    )


if __name__ == "__main__":
    main()
