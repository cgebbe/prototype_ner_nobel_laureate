"""
Based on https://huggingface.co/docs/transformers/task_summary#named-entity-recognition
"""
import functools

from datasets.load import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_dataset
import torch
import pandas as pd
import numpy as np

import utils

if 0:
    sequence = (
        "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, "
        "therefore very close to the Manhattan Bridge."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-cased",
        # "bert-base-cased",
        cache_dir=".cache",
    )
    tokens = tokenizer(sequence, return_tensors="pt")
    # tokens is a BatchEncoding and has keys
    # - input_ids = number for each token
    # - token_type_ids = 0 for each token?
    # - attention_mask = 1 for each token (since unpadded...)
    # tokens.tokens() = '[CLS]', 'Hu', '##gging', 'Face', 'Inc', '.', 'is', 'a', 'company', 'based', 'in', 'New', 'York', 'City', ...
    # values are tensors of shape (1,32)
elif 1:
    num_items = 8
    raw_datasets = load_dataset(
        "dataset.py",
        # "conll2003",
        cache_dir=".cache",
    )
    ds = raw_datasets.map(
        functools.partial(utils.preprocess, padding=False),
        batched=True,
    )
    subset = ds["test"].shuffle(seed=42)  # .select(range(num_items))
    item = subset[0]
    text = item["tokens"]

    tokens = {
        k: torch.Tensor([v]).long()
        for k, v in item.items()
        if k in ["input_ids", "attention_mask"]
    }

if 0:
    # from huggingface hub
    model = AutoModelForTokenClassification.from_pretrained(
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        cache_dir=".cache",
    )
elif 1:
    # from pretrained
    model = AutoModelForTokenClassification.from_pretrained(
        "output/20220109_095156/checkpoint-20",
        cache_dir=".cache",
    )

dct = dict(tokens)  # input_ids, token_type_ids, attention_mask
outputs = model(**dct).logits
predictions = torch.argmax(outputs, dim=2)

if 1:
    y_pred = predictions[0]
    y_true = item["labels"]
    # item["tokens"] has the words unsplitted in contrast to item["input_ids"]
    tokens = [utils.tokenizer.decode(x) for x in item["input_ids"]]
    assert len(y_pred) == len(y_true) == len(tokens)

    labels = subset.features["ner_tags"].feature.names
    labels_true = utils.convert_classes_to_labels([y_true], [y_true], labels)[0]
    labels_pred = utils.convert_classes_to_labels([y_pred], [y_true], labels)[0]
    valid_tokens = [t for t, l in zip(tokens, y_true) if l >= 0]
    assert len(labels_true) == len(labels_pred) == len(valid_tokens)

    df = pd.DataFrame(
        {"text": valid_tokens, "labels_true": labels_true, "labels_pred": labels_pred}
    )

    np.testing.assert_equal(df["labels_true"].values, df["labels_pred"].values)
    with pd.option_context("display.max_rows", 0):
        print(df)
