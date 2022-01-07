from datasets import load_dataset
from pprint import pprint

ds = load_dataset("conll2003", cache_dir=".cache")

ds = load_dataset("dataset.py", cache_dir=".cache")
subset = ds["train"]

num_items = len(subset)
item = subset[0]
pprint(item)
