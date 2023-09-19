import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

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

class ModelArguments():
    def __init__(self) -> None:
        pass

def evaluate(dataset, outputs, predictions):
    # save predictions
    with open(f"{DATASET}_predictions.txt", "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")

    # save original outputs
    with open(f"{DATASET}_outputs.txt", "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output + "\n")

    # calculate accuracy
    correct = 0
    for i in range(len(outputs)):
        if outputs[i] == dataset.labels[i]:
            correct += 1
    print("Accuracy: ", correct / len(outputs))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)