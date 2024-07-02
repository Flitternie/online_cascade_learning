import torch
from typing import Any, Tuple

class ModelWrapper(torch.nn.Module):
    """
    A wrapper class for a machine learning model with additional layers for further processing.

    Attributes:
        model (Any): The base model to be wrapped.
        device (torch.device): The device to run the model on (e.g., CPU or GPU).
        linear1 (torch.nn.Linear): The first linear layer.
        relu1 (torch.nn.ReLU): The ReLU activation function.
        linear2 (torch.nn.Linear): The second linear layer.
        sigmoid (torch.nn.Sigmoid): The Sigmoid activation function.
        name (str): The name of the model.
        learning_rate (float): The learning rate for training.
        regularization (float): The regularization parameter.
        decaying_factor (float): The decaying factor for jumping probability decay (see cascade for details).
        calibration (Any): The calibration parameter (see cascade for details).
    """

    def __init__(self, model: Any, args: Any) -> None:
        """
        Initialize the ModelWrapper with the base model and additional layers.

        Args:
            model (Any): The base model to be wrapped.
            args (Any): An argument object containing model and training parameters.
        """
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
    
    def forward(self, x: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the model and additional layers.

        Args:
            x (str): The input text data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The final output tensor and the intermediate probability tensor.
        """
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
    """
    A direct wrapper class for a machine learning model without additional layers.

    Attributes:
        model (Any): The base model to be wrapped.
        device (torch.device): The device to run the model on (e.g., CPU or GPU).
    """

    def __init__(self, model: Any, args: Any) -> None:
        """
        Initialize the ModelDirectWrapper with the base model.

        Args:
            model (Any): The base model to be wrapped.
            args (Any): An argument object containing model and training parameters.
        """
        super().__init__()
        self.model = model
        self.device = args.device
    
    def forward(self, x: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the base model.

        Args:
            x (str): The input text data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The probability tensor.
        """
        with torch.no_grad():
            probs = self.model.predict(x)
            probs = torch.Tensor(probs).to(self.device)
        return probs
