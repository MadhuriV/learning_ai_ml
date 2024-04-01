from transformers import pipeline

classifier = pipeline('sentiment-analysis')
response = classifier('I dont appreciate that tone')
print(response)