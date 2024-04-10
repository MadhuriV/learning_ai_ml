from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
response = generator(
    "Generate a list of three made-up book titles along with their authors and genres"
)

print(response)

