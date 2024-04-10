from transformers import AutoModel, AutoTokenizer
model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

local_model_directory="." # mention the localtion where you want to save the model
model.save_pretrained("local_model_directory")
tokenizer.save_pretrained("local_model_directory")

model = AutoModel.from_pretrained("local_model_directory")
tokenizer = AutoTokenizer.from_pretrained("local_model_directory")

inputs = tokenizer.encode("rephrase like a data analyst. what is MTD sales in Singapore", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))