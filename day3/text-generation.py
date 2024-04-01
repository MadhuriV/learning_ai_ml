from transformers import pipeline

generator = pipeline('text-generation', model='distilgpt2')

response = generator('In this course, we will teach you ',
                     max_length=30,
                     num_return_sequences=2)
print(response)