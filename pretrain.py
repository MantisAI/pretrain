import random
import json

from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm
import torch
import typer


app = typer.Typer()

def load_data(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            data.append(item["text"])
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mask_percentage=0.15):
        self.data = data
        self.tokenizer = tokenizer
        self.mask_percentage = mask_percentage

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx], padding="max_length", truncation=True)
        masked_input_ids = torch.tensor(
            [
                self.tokenizer.mask_token_id
                if (random.random() < self.mask_percentage)
                and (input_id != self.tokenizer.pad_token_id)
                else input_id
                for input_id in inputs["input_ids"]
            ]
        )
        labels = torch.tensor(inputs["input_ids"])
        return masked_input_ids, labels


@app.command()
def pretrain(
    data_path,
    model_path,
    pretrained_model="distilbert-base-uncased",
    batch_size:int=32,
    learning_rate:float=1e-5,
    epochs:int=5,
    mask_percentage:float=0.15,
    dry_run:bool=False
):
    """
    data_path: pure text, one document per line
    model_path: path to save model
    pretrained_model: name of pretrained model to use
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # load data
    data = load_data(data_path)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # pass to dataset and dataloader
    dataset = Dataset(data, tokenizer, mask_percentage)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load model to train
    model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
    model.to(device)

    # criterion, optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # torch training loop
    for epoch in range(epochs):
        batches = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in batches:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            preds = torch.transpose(outputs.logits, 2, 1)

            loss = criterion(preds, labels)
            loss.backward()

            optimizer.step()

            batches.set_postfix({"loss": loss.item()})

            if dry_run:
                break
        
        if dry_run:
            break 

    # save model
    model.save_pretrained(model_path)

if __name__ == "__main__":
    app()
