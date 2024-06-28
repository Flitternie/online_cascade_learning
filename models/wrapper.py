
import torch

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.device = args.device
        self.linear1 = torch.nn.Linear(args.num_labels, 128)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        self.name = args.name
        self.learning_rate = args.wrapper.learning_rate
        self.regularization = args.wrapper.regularization
        self.decaying_factor = args.wrapper.decaying_factor
        self.calibration = args.wrapper.calibration
    
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
