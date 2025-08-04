import pandas as pd
import re
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from clize import run

def extract_named_examples(name, examples):
    named_examples = []
    for example in examples:
        position = 0
        while position < len(example):
            idx = example.find('*', position)
            if idx == -1:
                break
            # Count the number of consecutive asterisks
            num_asterisks = 1
            while idx + num_asterisks < len(example) and example[idx + num_asterisks] == '*':
                num_asterisks += 1
            # Find the closing asterisks of the same length
            start_content = idx + num_asterisks
            closing_pattern = '*' * num_asterisks
            end_idx = example.find(closing_pattern, start_content)
            if end_idx == -1:
                position = start_content
                continue
            # Extract the content between the asterisks
            content = example[start_content:end_idx]
            if name in content:
                # Extract text after the closing asterisks until the next same number of asterisks
                after_closing = end_idx + num_asterisks
                next_asterisks = example.find(closing_pattern, after_closing)
                if next_asterisks == -1:
                    extracted_text = example[after_closing:]
                else:
                    extracted_text = example[after_closing:next_asterisks]
                named_examples.append(extracted_text)
            # Move past the closing asterisks
            position = end_idx + num_asterisks
    return named_examples


from clize import run

def train_lora(
    csv_file: str = 'slack_conversations.csv',
    model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
    output_dir: str = "./lora_models",
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 1,
    learning_rate: float = 1e-5,
    max_seq_length: int = 4096,
    lora_r: int = 1,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    sample_names: str = "Johanna,Samantha,Helena,Lysander,Rohan,Lena,Zara,Kai"
):
    """
    Train LoRA adapters for multiple personas using a pre-trained language model.

    :param csv_file: Path to the CSV file containing conversation data.
    :param model_id: Identifier of the pre-trained model to use.
    :param output_dir: Directory to save the trained models.
    :param num_train_epochs: Number of training epochs.
    :param per_device_train_batch_size: Batch size per device during training.
    :param learning_rate: Learning rate for training.
    :param max_seq_length: Maximum sequence length for input texts.
    :param lora_r: Rank of the LoRA update matrices.
    :param lora_alpha: LoRA alpha parameter.
    :param lora_dropout: Dropout probability for LoRA layers.
    :param sample_names: Comma-separated list of names to train models for.
    """
    import pandas as pd
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import torch

    df = pd.read_csv(csv_file, sep='|')
    df = df.dropna()

    day_responses = [df[f"day{i}"].tolist() for i in range(1, 9)]
    sample_name_list = sample_names.split(',')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
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
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    for name in sample_name_list:
        all_data = []
        for day_data in day_responses[:4]:  # Using first 4 days of data
            all_data.extend(extract_named_examples(name, day_data))
        
        data = [{"text": s} for s in all_data]
        dataset = Dataset.from_list(data)

        args = TrainingArguments(
            output_dir=f"{output_dir}/rank{lora_r}-lora-{name}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate
        )
        
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            dataset_text_field='text',
            max_seq_length=max_seq_length,
        )

        trainer.train()

        pt_save_directory = f"{output_dir}/lora-{name}"
        trainer.model.save_pretrained(pt_save_directory)



from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
import pyreft
from datasets import Dataset
from clize import run

def train_reft(
    model_id: str = "https://huggingface.co/meta-llama/Llama-3.1-70B",
    output_dir: str = "./reft_models",
    max_seq_length: int = 2048,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    warmup_steps: int = 100,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.0,
    num_train_epochs: int = 5,
    lora_r: int = 1,
    reft_layer: int = 12,
    sample_names: str = "Johanna,Samantha,Helena,Lysander,Rohan,Lena,Zara,Kai",
    csv_file: str = 'slack_conversations.csv',
):
    """
    Train REFT models for multiple personas using a pre-trained language model.

    :param model_id: Identifier of the pre-trained model to use.
    :param output_dir: Directory to save the trained models.
    :param max_seq_length: Maximum sequence length for input texts.
    :param per_device_train_batch_size: Batch size per device during training.
    :param gradient_accumulation_steps: Number of steps to accumulate gradients.
    :param warmup_steps: Number of warmup steps for learning rate scheduler.
    :param learning_rate: Learning rate for training.
    :param weight_decay: Weight decay for optimizer.
    :param num_train_epochs: Number of training epochs.
    :param lora_r: Rank of the RefT update matrices.
    :param reft_layer: Layer to apply RefT.
    :param sample_names: Comma-separated list of names to train models for.
    :param csv_file: Path to the CSV file containing conversation data.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

    tokenizer.pad_token = tokenizer.eos_token

    prompt_no_input_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    sample_name_list = sample_names.split(',')

    for name in sample_name_list:
        df = pd.read_csv(csv_file, sep='|')
        df = df.dropna()
        # Now you can work with the DataFrame 'df' as usual
        day1_responses = df["day1"].tolist()
        day2_responses = df["day2"].tolist()
        day3_responses = df["day3"].tolist()
        day4_responses = df["day4"].tolist()
        day_1_data = extract_named_examples(name, day1_responses)
        day_2_data = extract_named_examples(name, day2_responses)
        day_3_data = extract_named_examples(name, day3_responses)
        day_4_data = extract_named_examples(name, day4_responses)
        day_1_data.extend(day_2_data+day_3_data+day_4_data)
        data = [{"text": s} for s in day_1_data]
        dataset = Dataset.from_list(data)
        data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer, model, [prompt_no_input_template % row["text"] for row in dataset], 
        [row["text"] for row in dataset])

        reft_config = pyreft.ReftConfig(representations={
            "layer": reft_layer, "component": "block_output",
            "low_rank_dimension": lora_r,
            "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
            low_rank_dimension=lora_r)})
        reft_model = pyreft.get_reft_model(model, reft_config)
        reft_model.set_device("cuda")
        reft_model.print_trainable_parameters()
                
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            bf16=True,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="paged_adamw_32bit",
            weight_decay=weight_decay,
            lr_scheduler_type="cosine",
            output_dir=output_dir,
            report_to=[]
        )
        
        trainer = pyreft.ReftTrainerForCausalLM(model=reft_model, tokenizer=tokenizer, args=training_args, **data_module)
        
        _ = trainer.train()
        reft_model.save(
    save_directory=f"{output_dir}/reft-{name}-rank{lora_r}-epoch{num_train_epochs}")


if __name__ == "__main__":
    run(train_lora, train_reft)