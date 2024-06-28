import random
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, random_split

# Custom object to parse YAML data
class YAMLObject:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = YAMLObject(value)
            elif isinstance(value, list):
                value = [YAMLObject(item) if isinstance(item, dict) else item for item in value]
            self.__setattr__(key, value)
    
    def serialize(self) -> dict:
        return self.__dict__

# Function to load YAML file and convert it to the custom object
def load_config(file_path: str) -> YAMLObject:
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return YAMLObject(data)

# Custom Dataset class to load data
class GenericDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.text = data['text']
        try:
            self.labels = data['label']
        except:
            self.labels = data['llm_label']
        try:
            self.llm_labels = data['llm_label']
        except:
            pass
        self.num_labels = len(set(self.labels))
    
    def __len__(self) -> int:
        return len(self.text)
    
    def __getitem__(self, idx: int) -> tuple[str, int]:
        try:
            return self.text[idx], self.labels[idx], self.llm_labels[idx]
        except:
            return self.text[idx], self.labels[idx]

# Custom argument class to store model arguments
class ModelArguments():
    def __init__(self):
        pass

# Custom wrapper class 
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.device = args.device
        self.linear1 = torch.nn.Linear(args.num_labels, 128)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x: str) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            probs = self.model.predict(x)
            probs = torch.Tensor(probs).to(self.device)
            features = probs
        output = self.linear1(features)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output, probs

# Custom wrapper class 
class ModelDirectWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.device = args.device
    
    def forward(self, x: str) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            probs = self.model.predict(x)
            probs = torch.Tensor(probs).to(self.device)
        return probs
    
# Function to update model arguments
def update_config(config1: ModelArguments, config2: ModelArguments) -> ModelArguments:
    config = config1
    for key in dir(config2):
        if key[0] != "_":
            setattr(config, key, getattr(config2, key))
    return config

# Function for calculating normalized entropy
def normalized_entropy(x: torch.Tensor, gamma: int = 1, eps: float = 1e-6) -> torch.Tensor:
    x = x + eps # add eps to avoid log(0)
    x = x / torch.sum(x, dim=-1, keepdim=True)
    return (-torch.sum(x * torch.log(x), dim=-1) / np.log(x.shape[-1])) ** gamma

# Function to set seed for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Function to sort dictionary by key
def sort_dict_by_key(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}

# Function to check if two floating point numbers are equal
def equal(a: int | float, b: int | float) -> bool:
    return abs(a - b) < 1e-6

# Function to calculate calibrated confidence
def calibrated_confidence(confidence: float, gamma: float) -> list[float]:
    calibrated_unsureness = max(gamma, 1 - confidence)
    calibrated_confidence = 1 - calibrated_unsureness
    return [calibrated_confidence, calibrated_unsureness]