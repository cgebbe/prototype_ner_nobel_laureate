from typing import List
import transformers

IGNORE_CLASS = -100

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "distilbert-base-cased", cache_dir=".cache"
)


def preprocess(item, padding="max_length"):
    # item has dict_keys(['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
    # each entry is a list of BATCH_SIZE length
    # 'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
    # 'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]
    # raw_dataset.features["ner_tags"].feature.names = ['O', 'B-PER', ...]
    tokens = tokenizer(
        item["tokens"],
        padding=padding,
        truncation=True,
        is_split_into_words=True,
    )

    # tokens has dict_keys(['input_ids', 'token_type_ids', 'attention_mask']),
    # but is missing 'labels'!
    labels = []
    num_batches = len(tokens["input_ids"])
    for idx_batch in range(num_batches):
        token_labels = _convert_wordlabels_to_tokenlabels(
            word_labels=item["ner_tags"][idx_batch],
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
            token_labels.append(IGNORE_CLASS)
        elif word_idx == previous_word_idx:
            # if word is split into multiple tokens, only label first token
            token_labels.append(IGNORE_CLASS)
        else:
            token_labels.append(word_labels[word_idx])
        previous_word_idx = word_idx
    assert len(token_labels) == len(word_idx_per_token)
    return token_labels


def convert_classes_to_labels(
    classes, classes_true, label_per_class
) -> List[List[str]]:
    """Converts class indexes to labels

    Args:
        classes (Tensor): (batch_size, token_length) tensor of class indexes
        classes_true (Tensor): (batch_size, token_length) tensor of true class indexes
        label_per_class (List[str]): list of labels per class index

    Return:
        List[List[str]]: Nested list of labels
    """
    return [
        [label_per_class[p] for (p, l) in zip(pred, true) if l != IGNORE_CLASS]
        for pred, true in zip(classes, classes_true)
    ]
