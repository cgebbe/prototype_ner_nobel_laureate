import json

path = (
    r"/mnt/sda1/projects/git/prototypes/202112_ner/data/NER_Einstein/labels_manual.json"
)
with open(path) as f:
    dct = json.load(f)

print(dct)
