from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

sequence = (
    "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, "
    "therefore very close to the Manhattan Bridge."
)


tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased",
    cache_dir=".cache",
)
inputs = tokenizer(sequence, return_tensors="pt")
# inputs is a BatchEncoding and has keys
# - input_ids = number for each token
# - token_type_ids = 0 for each token?
# - attention_mask = 1 for each token (since unpadded...)

model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    cache_dir=".cache",
)

dct = dict(inputs)
outputs = model(**dct).logits
predictions = torch.argmax(outputs, dim=2)
