import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def load_data(config, tokenizer, split="train[:1%]", streaming=True):
    dataset = load_dataset(config["data_path"], config["data_config"], cache_dir= config["cache_dir"], split=split, streaming=streaming)

    test_size = 0.02 
    if streaming:
        train_dataset = dataset.filter(lambda _, idx: idx % 98 != 0 and idx % 99 != 0, with_indices=True)
        val_dataset = dataset.filter(lambda _, idx: idx % 98 == 0 or idx % 99 == 0, with_indices=True)

        train_dataset = train_dataset.shuffle(buffer_size=10_000, seed=config["train_args"]["seed"])
        val_dataset = val_dataset.shuffle(buffer_size=10_000, seed=config["train_args"]["seed"])
    else:
        split_dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=config["train_args"]["seed"]) 
        train_dataset, val_dataset = split_dataset["train"], split_dataset["test"]


    max_length = config["max_length"]

    if streaming:
        ds_kwargs = {}
    else:
        ds_kwargs = {"num_proc": config["train_args"]["num_proc"]}

    print(train_dataset[1]["text"])
    train_dataset = train_dataset.map(lambda ele: tokenizer(ele["text"],
                                                            truncation=True,
                                                            padding="max_length",
                                                            max_length=max_length),
                                    batched=True,
                                    **ds_kwargs)

    val_dataset = val_dataset.map(lambda ele: tokenizer(ele["text"],
                                                        truncation=True,
                                                        padding="max_length",
                                                        max_length=max_length),
                                batched=True,
                                **ds_kwargs)


    train_dataset = train_dataset.with_format("torch")

    train_dl = DataLoader(
        train_dataset,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer, mlm=False,
        ),
        batch_size=config["train_args"]["per_device_train_batch_size"],
    )

    val_dataset = val_dataset.with_format("torch")

    val_dl = DataLoader(
        val_dataset,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer, mlm=False,
        ),
        batch_size=config["train_args"]["per_device_eval_batch_size"],
    )

    return train_dl, val_dl


def tokenize_pairwise_rewards(examples, tokenizer, max_length):
    tokenized = {"chosen_input_ids": [], "chosen_attention_mask": [],  "rejected_input_ids": [], "rejected_attention_mask": []}

    for prompt, chosen, rejected in zip(examples["prompt"], examples["positive"], examples["negative"]):
        chosen_tokens = tokenizer(prompt + " " + chosen + tokenizer.eos_token, 
                                  truncation=True, 
                                  padding="max_length", 
                                  max_length=max_length, 
                                  return_tensors="pt")
        rejected_tokens = tokenizer(prompt + " " + rejected + tokenizer.eos_token,
                                    truncation=True, 
                                    padding="max_length", 
                                    max_length=max_length, 
                                    return_tensors="pt")

        tokenized["chosen_input_ids"].append(chosen_tokens["input_ids"])
        tokenized["chosen_attention_mask"].append(chosen_tokens["attention_mask"])
        
        tokenized["rejected_input_ids"].append(rejected_tokens["input_ids"])
        tokenized["rejected_attention_mask"].append(rejected_tokens["attention_mask"])

    return tokenized


def pairwise_collator(data):
    return {
        "chosen_input_ids": torch.cat([t["chosen_input_ids"] for t in data]),
        "chosen_attention_mask": torch.cat([t["chosen_attention_mask"] for t in data]),
        "rejected_input_ids": torch.cat([t["rejected_input_ids"] for t in data]),
        "rejected_attention_mask": torch.cat([t["rejected_attention_mask"] for t in data])
    }



def load_pairwise_reward_data(config, tokenizer, split="train", streaming=True):
    dataset = load_dataset(config["data_path"], split=split, streaming=streaming)

    max_length = config["max_length"]

    if streaming:
        ds_kwargs = {}
    else:
        ds_kwargs = {"num_proc": config["train_args"]["num_proc"]}

    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=config["train_args"]["seed"])
    train_dataset, val_dataset = split_dataset["train"], split_dataset["test"]

    

    train_dataset = train_dataset.map(lambda ele: tokenize_pairwise_rewards(ele, tokenizer, max_length),
                                                    batched=True,
                                                    remove_columns=["prompt", "positive", "negative", "timestep", "key"],
                                                    **ds_kwargs
                                                    )
    
    val_dataset = val_dataset.map(lambda ele: tokenize_pairwise_rewards(ele, tokenizer, max_length),
                                                    batched=True,
                                                    remove_columns=["prompt", "positive", "negative", "timestep", "key"],
                                                    **ds_kwargs
                                                    )

    
    train_dataset = train_dataset.with_format("torch")

    train_dl = DataLoader(
        train_dataset,
        collate_fn=pairwise_collator,
        batch_size=config["train_args"]["per_device_train_batch_size"],
        # have to drop last as we assume same batch size!
        drop_last=True
    )

    val_dataset = val_dataset.with_format("torch")

    val_dl = DataLoader(
        val_dataset,
        collate_fn=pairwise_collator,
        batch_size=config["train_args"]["per_device_eval_batch_size"],
        # have to drop last as we assume same batch size!
        drop_last=True
    )

    return train_dl, val_dl