import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse

import datasets
from torch.utils.data import Dataset, DataLoader, random_split

from utils import GenericDataset, set_seed

class BertModel():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, num_labels=self.args.num_labels)
        self.model = self.model.to('cuda')
        self.args.max_length = self.tokenizer.model_max_length
        print("BERT Model loaded")

    def train(self, train_dataloader, val_dataloader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        best_val_acc = 0
        for epoch in range(args.epochs):
            print("Epoch: ", epoch)
            self.model.train()
            # setup progress bar to show loss
            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to('cuda')
                true_labels = batch[1].to('cuda')
                outputs = self.model(**encoded_text, labels=true_labels)
                loss = outputs.loss
                pbar.set_description(f"Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # do evaluation
            predictions, val_acc = self.inference(args, val_dataloader, self.model, self.tokenizer)
            # save best performing model
            print("Validation Accuracy: ", val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("Saving model...")
                # create directory if it doesn't exist
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(self.model.state_dict(), f'models/bert_{self.args.data_card}.pt')

    def train_online(self, data):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.train()
        # setup progress bar to show loss
        pbar = tqdm(data)
        for step, batch in enumerate(pbar):
            encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
            encoded_text = encoded_text.to('cuda')
            true_labels = batch[1].to('cuda')
            outputs = self.model(**encoded_text, labels=true_labels)
            loss = outputs.loss
            pbar.set_description(f"Loss: {loss.item():.4f}")
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # torch.cuda.empty_cache()
        self.model.eval()

            

    def inference(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        outputs = torch.tensor([]).to('cuda')
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to('cuda')
                # compute accuracy
                logits = self.model(**encoded_text).logits
                predictions = torch.argmax(logits, dim=-1)
                true_labels = batch[1].to('cuda')
                correct += (predictions == true_labels).sum().item()
                total += len(true_labels)
                outputs = torch.cat((outputs, predictions))
        return outputs, correct / total

    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            input = self.tokenizer.encode_plus(input, padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
            input = input.to('cuda')
            output = self.model(**input)
            prediction = torch.argmax(output.logits, dim=-1).to('cpu').tolist()[0]
            # softmax
            confidence = torch.nn.functional.softmax(output.logits, dim=-1).to('cpu').tolist()[0]
        return prediction, confidence

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
    bert_model = BertModel(args)
    print("Model initialized")
    
    # train model
    if args.train:
        bert_model.train(args, train_dataloader, val_dataloader)
        print("Model trained")

    # inference with best model
    if args.inference:
        bert_model.model.load_state_dict(torch.load(f'models/bert_{args.data_card}.pt'))
        predictions, acc = bert_model.inference(args, test_dataloader)
        print("Inference done")
        print("Test Accuracy: ", acc)
        predictions = predictions.cpu().numpy()

        true_labels = np.array([x[1] for x in test_data])
        # calculate accuracy per class
        for i in range(dataset.num_labels):
            idx = (true_labels == i)
            print("Class: ", i, " Accuracy: ", np.mean(predictions[idx] == true_labels[idx]))
    
        with open(f'models/bert_{args.data_card}_outputs.txt', 'w') as f:
            for pred in predictions:
                f.write(str(int(pred)) + '\n')

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