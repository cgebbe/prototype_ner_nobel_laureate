"""
Train example from https://huggingface.co/docs/transformers/training
"""
import transformers
from datasets import load_dataset
from torch import nn

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "bert-base-cased", cache_dir=".cache"
)


IGNORE_LABEL = -100


def preprocess(data_item):
    """

    Output needs
    - input_ids
    - token_type_ids ?!
    - attention_mask
    - labels
    """
    # data_item has dict_keys(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
    # each entry is a list of BATCH_SIZE length
    tokens = tokenizer(
        data_item["tokens"],
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
    )
    # tokens has dict_keys(['input_ids', 'token_type_ids', 'attention_mask']).

    labels = []
    num_batches = len(tokens["input_ids"])
    for idx_batch in range(num_batches):
        token_labels = _convert_wordlabels_to_tokenlabels(
            word_labels=data_item["ner_tags"][idx_batch],
            word_idx_per_token=tokens.word_ids(batch_index=idx_batch),
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


raw_datasets = load_dataset("conll2003", cache_dir=".cache")
ds = raw_datasets.map(preprocess, batched=True, batch_size=100)
ds_train = ds["train"].shuffle(seed=42).select(range(2))
ds_eval = ds["test"].shuffle(seed=42).select(range(2))

features = raw_datasets["train"].features
labels = features["ner_tags"].feature.names

model = transformers.AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(labels), cache_dir=".cache"
)

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"], # (8,512)
        )
        outputs.logits.shape # 8,512,2
        inputs["labels"].shape # 8,512
        # output logits (8,2), inputs["labels"] has shape (8,512)
        loss = nn.BCEWithLogitsLoss()(outputs["logits"], inputs["labels"])
        return (loss, outputs) if return_outputs else loss


training_args = transformers.TrainingArguments("test_trainer")
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    # eval_dataset=ds_eval,
)
ret = trainer.train()
d = 0
