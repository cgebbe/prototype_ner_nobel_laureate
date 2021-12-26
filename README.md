# Goal: train a simple NER task

- [ ] overfit on single batch
  - [x] gather data
  - [ ] label manually - which format?
  - [ ] preprocess and train
  - [ ] 


## Sources

- simple NER pipeline
  - https://huggingface.co/dslim/bert-base-NER
- simple 
  - https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition
- NER = Token-classification, ahhhh!!
  - https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification
  - token classification = 
    - parts-of-speech tagging (POS) -> classify as VERB, NOUN, ...
    - named entity recognition (NER) -> classify as PERSON, LOC, ...
    - phrase extraction (CHUNKS)
    - see https://stackabuse.com/python-for-nlp-parts-of-speech-tagging-and-named-entity-recognition/

# Questions

## How to run NER training using run_ner.py

```bash
pip install datasets
pip install seqeval  # required for evaluating

python3 run_ner.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name conll2003 \
  --output_dir /tmp/test-ner \
  --do_train \
  --do_eval
```

## How does custom dataset need to look like? (instead of conll2003?)

- https://huggingface.co/docs/datasets/loading_datasets.html
  - does huggingface use IOB or IOB2 format?

## How to compute metric?

- see https://huggingface.co/metrics/seqeval 
- forwards to https://github.com/chakki-works/seqeval

```python
>>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> f1_score(y_true, y_pred)
0.50
```

## How does head of CNN look like (labels -> softmax?)

- simply parse num_labels, id2label, label2id

## How to label using doccano?

https://github.com/doccano/doccano

```bash
# create container
docker pull doccano/doccano
docker container create --name doccano \
  -e "ADMIN_USERNAME=admin" \
  -e "ADMIN_EMAIL=admin@example.com" \
  -e "ADMIN_PASSWORD=password" \
  -p 8000:8000 doccano/doccano

# start container, go to http://127.0.0.1:8000/
docker container start doccano

# to stop
docker container stop doccano -t 5
```

### Exported text file is binary :/

- https://github.com/doccano/doccano/issues/1606
- Workaround: Simply create manually using JSONL

```json
{"text": "EU rejects German call to boycott British lamb.", "label": [ [0, 2, "ORG"], [11, 17, "MISC"], ... ]}
{"text": "Peter Blackburn", "label": [ [0, 15, "PERSON"] ]}
{"text": "President Obama", "label": [ [10, 15, "PERSON"] ]}
```


# How to setup project next time

- setup DOCKERFILE
  - use colored terminal https://stackoverflow.com/a/33499558/2135504
  - use own ID https://issueexplorer.com/issue/microsoft/vscode-remote-release/5542
- setup README.md
- git init -> first commit
- setup .vscode
  - use dockerfile
  - install python
  - git?! - always not sure whether from inside or outside container...

- also, when open in docker-remote cannot open locally?! check permissions... 