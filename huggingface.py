from datasets import list_datasets, load_dataset

# List all the datasets in your cache
print(list_datasets())

# Load a dataset from your cache
dataset = load_dataset("pile-of-law/pile-of-law","atticus_contracts")