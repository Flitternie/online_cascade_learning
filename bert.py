import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from tqdm import tqdm
import argparse

import datasets
from torch.utils.data import Dataset, DataLoader, random_split

from utils import *

class BertModel():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, num_labels=self.args.num_labels)
        self.model = self.model.to('cuda')
        self.args.max_length = self.tokenizer.model_max_length

        if "class_weight" in dir(self.args):
            self.class_weight = self.args.class_weight
            self.class_count = [0 for _ in range(self.args.num_labels)]
        print("BERT Model loaded")
    
    def initialize_model(self):
        # clean up GPU memory and reload model
        torch.cuda.empty_cache()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, num_labels=self.args.num_labels)
        self.model = self.model.to('cuda')

    def train(self, train_dataloader, val_dataloader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        best_val_acc = 0
        for epoch in range(self.args.epochs):
            print("Epoch: ", epoch+1)
            self.model.train()
            if "class_weight" in dir(self.args):
                if self.args.class_weight == "balanced":
                    for batch in train_dataloader:
                        for label in batch[1]:
                            self.class_count[int(label)] += 1
                    # balanced class weights computed by: n_samples / (n_classes * np.bincount(y))
                    self.class_weight =  {i: sum(self.class_count) / max(( self.args.num_labels * self.class_count[i] ), 1) for i in range(self.args.num_labels)}
                    # normalize to sum up to 1
                    self.class_weight = {k: v / sum(self.class_weight.values()) for k, v in self.class_weight.items()}    
                assert isinstance(self.class_weight, dict), f"Class weight is not a dict: {self.class_weight}"
                assert len(self.class_weight.keys()) == self.args.num_labels, f"Class weight keys: {self.class_weight.keys()} not equal to num_labels: {self.args.num_labels}"
                assert equal(sum(self.class_weight.values()), 1), f"Sum of class weights is {sum(self.class_weight.values())}"
                self.class_weight = sort_dict_by_key(self.class_weight)
            else:
                self.class_weight = None
                
            # setup progress bar to show loss
            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to('cuda')
                true_labels = batch[1].to('cuda')
                outputs = self.model(**encoded_text, labels=true_labels)
                if self.class_weight is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(list(self.class_weight.values()), dtype=torch.float32).to('cuda'))
                    loss = loss_fct(outputs.logits.view(-1, self.args.num_labels), true_labels.view(-1))
                else:
                    loss = outputs.loss
                pbar.set_description(f"Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # do evaluation
            predictions, val_acc = self.inference(val_dataloader)
            # save best performing model
            print("Validation Accuracy: ", val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print("Saving model...")
                # create directory if it doesn't exist
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(self.model.state_dict(), f'{self.args.model_dir}.pt')

    def train_online(self, data):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.model.train()
        if "class_weight" in dir(self.args):
            if self.args.class_weight == "balanced":
                for batch in data:
                    for label in batch[1]:
                        self.class_count[int(label)] += 1
                # balanced class weights computed by: n_samples / (n_classes * np.bincount(y))
                self.class_weight =  {i: sum(self.class_count) / max(( self.args.num_labels * self.class_count[i] ), 1) for i in range(self.args.num_labels)}
                # normalize to sum up to 1
                self.class_weight = {k: v / sum(self.class_weight.values()) for k, v in self.class_weight.items()}    
            assert isinstance(self.class_weight, dict), f"Class weight is not a dict: {self.class_weight}"
            assert len(self.class_weight.keys()) == self.args.num_labels, f"Class weight keys: {self.class_weight.keys()} not equal to num_labels: {self.args.num_labels}"
            assert equal(sum(self.class_weight.values()), 1), f"Sum of class weights is {sum(self.class_weight.values())}"
            self.class_weight = sort_dict_by_key(self.class_weight)
        else:
            self.class_weight = None
        # setup progress bar to show loss
        pbar = tqdm(data)
        for step, batch in enumerate(pbar):
            encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
            encoded_text = encoded_text.to('cuda')
            true_labels = batch[1].to('cuda')
            outputs = self.model(**encoded_text, labels=true_labels)
            if "class_weight" in dir(self.args):
                loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(list(self.class_weight.values()), dtype=torch.float32).to('cuda'))
                loss = loss_fct(outputs.logits.view(-1, self.args.num_labels), true_labels.view(-1))
            else:
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
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print("Data loaded, train size: ", len(train_data), " val size: ", len(val_data), " test size: ", len(test_data))
    
    # initialize model
    args.num_labels = dataset.num_labels
    bert_model = BertModel(args)
    print("Model initialized")
    
    # train model
    if args.train:
        bert_model.train(train_dataloader, val_dataloader)
        print("Model trained")

    # inference with best model
    if args.inference:
        bert_model.model.load_state_dict(torch.load(f'{args.model_dir}.pt'))
        predictions, acc = bert_model.inference(test_dataloader)
        print("Inference done")
        print("Test Accuracy: ", acc)
        predictions = predictions.cpu().numpy()

        true_labels = np.array([x[1] for x in test_data])
        # calculate accuracy per class
        for i in range(dataset.num_labels):
            idx = np.where(true_labels == i)[0]
            print("Class: ", i, " Accuracy: ", np.sum(predictions[idx] == true_labels[idx]) / len(idx))
    
        with open(f'{args.model_dir}_outputs.txt', 'w') as f:
            for pred, gold in zip(predictions, true_labels):
                f.write(str(int(pred)) + '\t' + str(int(gold)) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_card', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)
    main(args)