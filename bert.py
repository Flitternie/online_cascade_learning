import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse

import datasets
from torch.utils.data import Dataset, DataLoader, random_split

from utils import set_seed

class GenericDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.text = data['text']
        self.labels = data['label']
        self.num_labels = len(set(self.labels))
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]


def initialize_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
    model = model.to('cuda')
    return model, tokenizer

def train(args, model, tokenizer, train_dataloader, val_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_val_acc = 0
    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        model.train()
        # setup progress bar to show loss
        pbar = tqdm(train_dataloader)
        for step, batch in enumerate(pbar):
            encoded_text = tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=args.max_length, truncation=True, return_tensors='pt')
            encoded_text = encoded_text.to('cuda')
            true_labels = batch[1].to('cuda')
            outputs = model(**encoded_text, labels=true_labels)
            loss = outputs.loss
            pbar.set_description(f"Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # do evaluation
        predictions, val_acc = inference(args, val_dataloader, model, tokenizer)
        # save best performing model
        print("Validation Accuracy: ", val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Saving model...")
            # create directory if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), f'models/bert_{args.data_card}.pt')
    return model

def inference(args, dataloader, model, tokenizer):
    model.eval()
    correct, total = 0, 0
    outputs = torch.tensor([]).to('cuda')
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            encoded_text = tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=args.max_length, truncation=True, return_tensors='pt')
            encoded_text = encoded_text.to('cuda')
            # compute accuracy
            logits = model(**encoded_text).logits
            predictions = torch.argmax(logits, dim=-1)
            true_labels = batch[1].to('cuda')
            correct += (predictions == true_labels).sum().item()
            total += len(true_labels)
            outputs = torch.cat((outputs, predictions))
    return outputs, correct / total

def main(args):
    data = datasets.load_from_disk(args.data_dir)
    
    # split into train, val, test
    dataset = GenericDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
    print("Data loaded, train size: ", len(train_data), " val size: ", len(val_data), " test size: ", len(test_data))
    
    # initialize model
    args.num_labels = dataset.num_labels
    model, tokenizer = initialize_model(args)
    args.max_length = tokenizer.model_max_length
    print("Model initialized")
    
    # train model
    if args.train:
        model = train(args, model, tokenizer, train_dataloader, val_dataloader)
        print("Model trained")

    # inference with best model
    if args.inference:
        model.load_state_dict(torch.load(f'models/bert_{args.data_card}.pt'))
        predictions, acc = inference(args, test_dataloader, model, tokenizer)
        print("Inference done")
        print("Test Accuracy: ", acc)
        predictions = predictions.cpu().numpy()

        true_labels = np.array([x[1] for x in test_data])
        # calculate accuracy per class
        for i in range(dataset.num_labels):
            idx = (true_labels == i)
            print("Class: ", i, " Accuracy: ", (predictions[idx] == true_labels[idx]).sum() / len(true_labels[idx]))
    
        with open('models/bert_outputs.txt', 'w') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_card', type=str, default='agnews')
    parser.add_argument('--data_dir', type=str, default='data/2000_sampled_agnews')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)