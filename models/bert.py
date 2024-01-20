import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import *

class BertModel():
    def __init__(self, args: ModelArguments) -> None:
        self.args = args
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, num_labels=self.args.num_labels)
        self.model = self.model.to(self.device)
        self.args.max_length = self.tokenizer.model_max_length
        self.online_cache = {"text": [], "llm_label": []}
        if "class_weight" in dir(self.args):
            self.class_weight = self.args.class_weight
            self.class_count = [0 for _ in range(self.args.num_labels)]
        print("BERT Model loaded")
    
    def initialize_model(self) -> None:
        # clean up GPU memory and reload model
        torch.cuda.empty_cache()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, num_labels=self.args.num_labels)
        self.model = self.model.to(self.device)
    
    def cache_add(self, text: str, label: int) -> None:
        self.online_cache["text"].append(text)
        self.online_cache["llm_label"].append(label)
    
    def cache_clear(self) -> None:
        self.online_cache = {"text": [], "llm_label": []}

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        best_val_acc = 0
        for epoch in range(self.args.num_epochs):
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
                encoded_text = encoded_text.to(self.device)
                true_labels = batch[-1].to(self.device)
                outputs = self.model(**encoded_text, labels=true_labels)
                if self.class_weight is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(list(self.class_weight.values()), dtype=torch.float32).to(self.device))
                    loss = loss_fct(outputs.logits.view(-1, self.args.num_labels), true_labels.view(-1))
                else:
                    loss = outputs.loss
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # do evaluation
        predictions, val_acc = self.evaluate(val_dataloader)
        # save best performing model
        print("Validation Accuracy: ", val_acc)
        # self.tokenizer.save_pretrained(f'{self.args.model_dir}')
        # self.model.save_pretrained(f'{self.args.model_dir}', from_pt=True)
        return val_acc

    def train_online(self, data: dict) -> None:
        online_data = GenericDataset(self.online_cache)
        data = DataLoader(online_data, batch_size=self.args.batch_size, shuffle=True)
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
        for epoch in range(self.args.num_epochs):
            for step, batch in enumerate(data):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to(self.device)
                true_labels = batch[1].to(self.device)
                outputs = self.model(**encoded_text, labels=true_labels)
                logits = nn.functional.softmax(outputs.logits, dim=-1)
                if "class_weight" in dir(self.args):
                    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(list(self.class_weight.values()), dtype=torch.float32).to(self.device))
                else:
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(outputs.logits.view(-1, self.args.num_labels), true_labels.view(-1))
                loss.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.model.eval()            

    def inference(self, dataloader: DataLoader) -> torch.tensor:
        self.model.eval()
        probs = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to(self.device)
                logits = self.model(**encoded_text).logits
                prob = nn.functional.softmax(logits, dim=-1)
                probs = torch.cat((probs, prob))
        return probs

    def evaluate(self, dataloader: DataLoader) -> tuple[torch.tensor, float]:
        correct, total = 0, 0
        outputs = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader)):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to(self.device)
                # compute accuracy
                logits = self.model(**encoded_text).logits
                predictions = torch.argmax(logits, dim=-1)
                true_labels = batch[1].to(self.device)
                correct += (predictions == true_labels).sum().item()
                total += len(true_labels)
                outputs = torch.cat((outputs, predictions))
        return outputs, correct / total
    
    def predict(self, input: str) -> torch.tensor:
        self.model.eval()
        with torch.no_grad():
            input = self.tokenizer.encode_plus(input, padding='max_length', max_length=self.args.max_length, truncation=True, return_tensors='pt')
            input = input.to(self.device)
            logits = self.model(**input).logits
            probs = nn.functional.softmax(logits, dim=-1)
        return probs