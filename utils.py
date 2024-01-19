import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

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

class ModelArguments():
    def __init__(self):
        pass

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.linear1 = torch.nn.Linear(args.num_labels, 128)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x: str) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            probs = self.model.predict(x)
            probs = torch.Tensor(probs).to('cuda')
            features = probs
        output = self.linear1(features)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output, probs

# class ModelWrapper(torch.nn.Module):
#     def __init__(self, model, args):
#         super().__init__()
#         self.model = model
#         self.linear1 = torch.nn.Linear(1 + args.num_labels + args.num_labels, 2**6)
#         self.relu1 = torch.nn.ReLU()
#         self.linear2 = torch.nn.Linear(2**6, 2**4)
#         self.relu2 = torch.nn.ReLU()
#         self.linear3 = torch.nn.Linear(2**4, 1)
#         self.sigmoid = torch.nn.Sigmoid()
    
#     def forward(self, x: str) -> tuple[torch.Tensor, torch.Tensor]:
#         with torch.no_grad():
#             probs = self.model.predict(x)
#             probs = torch.Tensor(probs).to('cuda')
#             normalized_entropy_probs = torch.Tensor(normalized_entropy(probs)).unsqueeze(-1).to('cuda')
#             one_hot_argmax_probs = torch.nn.functional.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.shape[-1]).float().to('cuda')
#             features = torch.cat((normalized_entropy_probs, probs, one_hot_argmax_probs), dim=-1)
#         output = self.linear1(features)
#         output = self.relu1(output)
#         output = self.linear2(output)
#         output = self.relu2(output)
#         output = self.linear3(output)
#         output = self.sigmoid(output)
#         return output, probs

def update_config(config1: ModelArguments, config2: ModelArguments) -> ModelArguments:
    config = config1
    for key in dir(config2):
        if key[0] != "_":
            setattr(config, key, getattr(config2, key))
    return config

def normalized_entropy(x: torch.Tensor, gamma: int = 1, eps: float = 1e-6) -> torch.Tensor:
    x = x + eps # add eps to avoid log(0)
    x = x / torch.sum(x, dim=-1, keepdim=True)
    return (-torch.sum(x * torch.log(x), dim=-1) / np.log(x.shape[-1])) ** gamma

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sort_dict_by_key(d: dict) -> dict:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}

def equal(a: int | float, b: int | float) -> bool:
    return abs(a - b) < 1e-6

def calibrated_confidence(confidence: float, gamma: float) -> list[float]:
    calibrated_unsureness = max(gamma, 1 - confidence)
    calibrated_confidence = 1 - calibrated_unsureness
    return [calibrated_confidence, calibrated_unsureness]