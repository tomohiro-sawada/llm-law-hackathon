from itertools import chain
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import load_yaml
import argparse

# dataloaders


def build_dataloaders(
    config,
    sequence_length: int = 8192,
):
    """
    Build data loaders for training.

    This function performs the following steps:
    1. Load the tokenizer from the pretrained "EleutherAI/gpt-neox-20b" model.
    2. Load the "openwebtext" dataset.
    3. Tokenize the dataset, adding the end-of-sentence token to each text.
    4. Process the tokenized dataset into chunks of a specified block size.

    Returns:
        Dataset: The processed dataset ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    dataset = load_dataset(config["data_path"], split="train[:1%]")
    dataset = dataset.shuffle()

    tokenized_dataset = dataset.map(
        lambda example: tokenizer([t + tokenizer.eos_token for t in example["text"]]),
        batched=True,
        remove_columns=["text"],
        num_proc=32,
    )

    block_size = sequence_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    train_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=32,
    )

    train_dataset.push_to_hub(config["savedata_dir"], private=True)
    return train_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    # parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config_path)

    build_dataloaders(config)