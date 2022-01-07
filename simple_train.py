"""
Train example from https://huggingface.co/docs/transformers/training
"""
import transformers
from datasets import load_dataset, load_metric
from pprint import pprint
import numpy as np
import datetime
import pandas as pd

import utils


def create_metric_function(labels):
    """Creates metric

    Args:
        labels (List[str]): e.g. ['O', 'B_PER', 'I_PER']

    Returns:
        Callable[EvaluationOutput, Dict]: function to compute metric
    """
    # seqeval directly supports IOB, IOB2, see https://github.com/chakki-works/seqeval
    metric = load_metric("seqeval")

    def compute_metrics(p):
        y_true = p.label_ids
        y_pred = np.argmax(p.predictions, axis=2)  # (batch_size,512,num_labels)
        assert y_pred.shape == y_true.shape  # (batch_size,512)

        labels_true = utils.convert_classes_to_labels(y_true, y_true, labels)
        labels_pred = utils.convert_classes_to_labels(y_pred, y_true, labels)

        # understand KPI calculation
        if 0:
            _flatten_lst = lambda lst: [item for sublist in lst for item in sublist]
            df = pd.DataFrame(
                {
                    "true": _flatten_lst(labels_true),
                    "pred": _flatten_lst(labels_pred),
                }
            )

        results = metric.compute(
            predictions=labels_pred,
            references=labels_true,
            scheme="IOB2",  # see https://huggingface.co/datasets/conll2003
            mode="strict",
            # zero_division='',  # TODO, check seqeval
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


num_items = 8
raw_datasets = load_dataset(
    "dataset.py",
    # "conll2003",
    cache_dir=".cache",
)
ds = raw_datasets.map(utils.preprocess, batched=True)
ds_train = ds["train"].shuffle(seed=42).select(range(num_items))
ds_eval = ds["test"].shuffle(seed=42).select(range(num_items))

labels = raw_datasets["train"].features["ner_tags"].feature.names

# we could also limit the model to a sequence length of 256, but then couldn't load the pretrained weights?!
model = transformers.AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=len(labels),
    cache_dir=".cache",
    # gradient_checkpointing only works for bert, not for distilbert
    # gradient_checkpointing=True,  # see BertConfig and https://github.com/huggingface/transformers/blob/0735def8e1200ed45a2c33a075bc1595b12ef56a/src/transformers/modeling_bert.py#L461
)

output_dir = f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
training_args = transformers.TrainingArguments(
    # --- how to train
    num_train_epochs=30,  # defaults to 3
    per_device_train_batch_size=1,  # defaults to 8
    gradient_accumulation_steps=8,  # defaults to 1
    # TODO: learning rate seems to decrease linearly :/
    # no_cuda=True,  # if GPU too small, see https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues
    # --- how to log
    output_dir=output_dir,
    logging_dir=output_dir + "/logs",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,  # delete any older checkpoint
)
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    compute_metrics=create_metric_function(labels),
    # callbacks=[transformers.integrations.TensorBoardCallback()],
)
train_output = trainer.train()
pprint(train_output.metrics)

evaluation_metrics = trainer.evaluate()
pprint(evaluation_metrics)

d = 0
