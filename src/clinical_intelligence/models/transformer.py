from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_sequence_classifier(checkpoint: str, labels: list[str]):
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    return tokenizer, model, label2id, id2label
