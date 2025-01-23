import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# Loading and preparing data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Converting a String to a List
    def safe_eval(value):
        try:
            return eval(value) if isinstance(value, str) and value.startswith(
                "[") and value.endswith("]") else []
        except:
            return []

    # Convert the "tokens" and "tags" columns to lists
    df["tokens"] = df["tokens"].apply(safe_eval)
    df["tags"] = df["tags"].apply(safe_eval)

    # # Casting tags to integer type with safe handling
    tag_map = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}

    def map_tags(tags):
        mapped_tags = []
        for tag in tags:
            if tag in tag_map:
                mapped_tags.append(tag_map[tag])
            else:
                mapped_tags.append(tag_map["O"])  # If the tag is not found, assign "O"
        return mapped_tags

    df["tags"] = df["tags"].apply(map_tags)

    return df

# Preparing data for the model
def prepare_data(df):
    tokenized_data = {"input_ids": [], "attention_mask": [], "labels": []}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    for tokens, tags in zip(df["tokens"], df["tags"]):
        tokenized = tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            is_split_into_words=True
        )
        labels = [-100] * len(tokenized["input_ids"])

        word_ids = tokenized.word_ids()
        previous_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id != previous_word_id:
                labels[idx] = tags[word_id]
            previous_word_id = word_id

        tokenized_data["input_ids"].append(tokenized["input_ids"])
        tokenized_data["attention_mask"].append(tokenized["attention_mask"])
        tokenized_data["labels"].append(labels)

    return Dataset.from_dict(tokenized_data)

# Тренировка модели
def train_model(train_dataset, test_dataset):
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased", num_labels=3
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=2,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./ner_model")
    tokenizer.save_pretrained("./ner_model")

    print("The model is trained and saved.")

    return model, tokenizer
