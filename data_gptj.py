import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, DefaultDataCollator
from random import shuffle

def load_data(config, tokenizer, split="train", streaming=True):
    dataset = load_dataset(config["data_path"], split=split, streaming=streaming,cache_dir=config["cache_dir"])

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

    train_dataset = train_dataset.map(lambda ele: tokenizer([t + "<|endoftext|>" for t in ele["response"]],
                                                            truncation=True,
                                                            padding="max_length",
                                                            max_length=max_length),
                                                            batched=True,
                                                            # remove_columns=["response", "prompt"],
                                                            **ds_kwargs)

    val_dataset = val_dataset.map(lambda ele: tokenizer([t + "<|endoftext|>" for t in ele["response"]],
                                                            truncation=True,
                                                            padding="max_length",
                                                            max_length=max_length),
                                                            batched=True,
                                                            # remove_columns=["response", "prompt"],
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