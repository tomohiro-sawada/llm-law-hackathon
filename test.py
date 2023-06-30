from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tsawada/llmlh")
model = AutoModel.from_pretrained("tsawada/llmlh")


input_text = "Hello, World!"
inputs = tokenizer(input_text, return_tensors="pt")


output = model(**inputs)
print(output)