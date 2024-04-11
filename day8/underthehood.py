from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

# preprocessing with tokenizers
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# passing the inputs through the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
print(outputs.logits)

# postprocessing
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)
