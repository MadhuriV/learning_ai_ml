from transformers import pipeline

classifier = pipeline('sentiment-analysis')
response = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(response)