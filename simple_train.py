"""
Train example from https://huggingface.co/docs/transformers/training
"""
import transformers
from datasets import load_dataset, load_metric
from pprint import pprint
import numpy as np
import datetime

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "distilbert-base-cased", cache_dir=".cache"
)
IGNORE_LABEL = -100


def preprocess(data_item):
    # data_item has dict_keys(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
    # each entry is a list of BATCH_SIZE length
    tokens = tokenizer(
        data_item["tokens"],
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
    )

    # tokens has dict_keys(['input_ids', 'token_type_ids', 'attention_mask']),
    # but is missing 'labels'!
    labels = []
    num_batches = len(tokens["input_ids"])
    for idx_batch in range(num_batches):
        token_labels = _convert_wordlabels_to_tokenlabels(
            word_labels=data_item["ner_tags"][idx_batch],
            word_idx_per_token=tokens[idx_batch].word_ids,
        )
        labels.append(token_labels)
    tokens["labels"] = labels
    assert len(labels) == num_batches
    return tokens


def _convert_wordlabels_to_tokenlabels(word_labels, word_idx_per_token):
    """Converts labels per word to labels per token

    Args:
        word_labels (List[int]): Labels per word
        word_idx_per_token (List[Optional[int]]): word index per token

    Returns:
        List[int]: labels per token
    """
    token_labels = []
    previous_word_idx = None
    for word_idx in word_idx_per_token:
        if word_idx is None:
            token_labels.append(IGNORE_LABEL)
        elif word_idx == previous_word_idx:
            # if word is split into multiple tokens, only label first token
            token_labels.append(IGNORE_LABEL)
        else:
            token_labels.append(word_labels[word_idx])
        previous_word_idx = word_idx
    assert len(token_labels) == len(word_idx_per_token)
    return token_labels


num_items = 20
raw_datasets = load_dataset("conll2003", cache_dir=".cache")
ds = raw_datasets.map(preprocess, batched=True)
ds_train = ds["train"].shuffle(seed=42).select(range(num_items))
ds_eval = ds["test"].shuffle(seed=42).select(range(num_items))

features = raw_datasets["train"].features
labels = features["ner_tags"].feature.names

# we could also limit the model to a sequence length of 256, but then couldn't load the pretrained weights
model = transformers.AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=len(labels),
    cache_dir=".cache",
    # gradient_checkpointing only works for bert, not for distilbert
    # gradient_checkpointing=True,  # see BertConfig and https://github.com/huggingface/transformers/blob/0735def8e1200ed45a2c33a075bc1595b12ef56a/src/transformers/modeling_bert.py#L461
)


def create_metric_function(labels, ignore_label):
    """Creates metric

    Args:
        labels (List[str]): e.g. ['O', 'B_PER', 'I_PER']
        IGNORE_LABEL (int): label ID to ignore

    Returns:
        Callable[EvaluationOutput, Dict]: function to compute metric
    """
    # seqeval directly supports IOB, IOB2, see https://github.com/chakki-works/seqeval
    metric = load_metric("seqeval")

    def compute_metrics(p):
        y_true = p.label_ids
        y_pred = np.argmax(p.predictions, axis=2)  # (batch_size,512,num_labels)
        assert y_pred.shape == y_true.shape  # (batch_size,512)

        # change to nested list and replace
        words_pred = [
            [labels[p] for (p, l) in zip(prediction, label) if l != ignore_label]
            for prediction, label in zip(y_pred, y_true)
        ]
        words_true = [
            [labels[l] for (p, l) in zip(prediction, label) if l != ignore_label]
            for prediction, label in zip(y_pred, y_true)
        ]

        results = metric.compute(
            predictions=words_pred,
            references=words_true,
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


output_dir = f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

training_args = transformers.TrainingArguments(
    # --- how to train
    num_train_epochs=3,  # defaults to 3
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
    compute_metrics=create_metric_function(labels, IGNORE_LABEL),
    # callbacks=[transformers.integrations.TensorBoardCallback()],
)
train_output = trainer.train()
pprint(train_output.metrics)

evaluation_metrics = trainer.evaluate()
pprint(evaluation_metrics)

d = 0
