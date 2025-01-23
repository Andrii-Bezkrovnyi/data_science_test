from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# Loading the model and tokenizer
def load_model(model_path="ner_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return model, tokenizer


# Initializing the pipeline for NER
def get_ner_pipeline(model, tokenizer):
    return pipeline("ner", model=model, tokenizer=tokenizer,
                    aggregation_strategy="simple")


# Converting Labels to Strings
def convert_labels(entities):
    label_map = {0: "O", 1: "B-MOUNTAIN", 2: "I-MOUNTAIN"}
    for entity in entities:
        entity['entity_group'] = label_map.get(
            int(entity['entity_group'].replace('LABEL_', '')), entity['entity_group'])
    return entities


# Function for performing inference
def run_inference(text, model_path="ner_model"):
    # Loading the model and tokenizer
    model, tokenizer = load_model(model_path)

    # Get a pipeline
    ner_pipeline = get_ner_pipeline(model, tokenizer)

    # Perform inference
    entities = ner_pipeline(text)

    # Transforming Labels
    entities = convert_labels(entities)

    return entities
