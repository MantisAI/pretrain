# Pretrain

A tool to pretrain neural network models before fine tuning on a
give task.

# Quickstart

Install the package
```
git clone https://github.com/MantisAI/pretrain.git
pip install .
```

Run pretrain
```
pretrain data.jsonl models/
```

You need data in a JSONL format with a `text` key

# Pretrain CLI

```
Usage: pretrain [OPTIONS] DATA_PATH MODEL_PATH

  data_path: pure text, one document per line model_path: path to save model
  pretrained_model: name of pretrained model to use

Arguments:
  DATA_PATH   [required]
  MODEL_PATH  [required]

Options:
  --pretrained-model TEXT         [default: distilbert-base-uncased]
  --batch-size INTEGER            [default: 32]
  --learning-rate FLOAT           [default: 1e-05]
  --epochs INTEGER                [default: 5]
  --mask-percentage FLOAT         [default: 0.15]
  --dry-run / --no-dry-run        [default: no-dry-run]
  --help                          Show this message and exit.
```

# Contribute

Create and activate a virtualenv e.g.
```
python -m venv venv
source venv/bin/activate
```

Install dependencies and package
```
pip install -r requirements.txt
pip install -e .
```

