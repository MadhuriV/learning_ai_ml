from transformers import pipeline

classifier = pipeline("zero-shot-classification")
response = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(response)
