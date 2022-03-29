import json
import os

import pytest

from pretrain import pretrain


@pytest.fixture
def data_path(tmp_path):
    data_path = os.path.join(tmp_path, "data.jsonl")

    data = [
        "This is the first test sentence",
        "Use that data to test pretrain"
    ] * 5
    with open(data_path, "w") as f:
        for text in data:
            f.write(json.dumps({"text": text})+"\n")
    return data_path

@pytest.fixture
def model_path(tmp_path):
    model_path = os.path.join(tmp_path, "models")
    return model_path

def test_pretrain(data_path, model_path):
    pretrain(data_path, model_path, dry_run=True)
    assert os.path.exists(os.path.join(model_path, "config.json"))
