    train_dataset = train_dataset.map(lambda ele: tokenizer(ele["text"],
                                                            truncation=True,
                                                            padding="max_length",
                                                            max_length=max_length),
                                    batched=True,
                                    **ds_kwargs)