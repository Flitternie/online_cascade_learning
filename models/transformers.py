import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.base_model import BaseModel
from utils import ModelArguments, GenericDataset

class GenericTransformersModel(BaseModel):
    """
    A generic model class for training and evaluating HuggingFace transformer models.

    Attributes:
        args (ModelArguments): Configuration arguments for the model.
        device (torch.device): The device to run the model on (CPU or GPU).
        tokenizer (AutoTokenizer): Tokenizer for the transformer model.
        model (AutoModelForSequenceClassification): The transformer model instance.
        class_count (list): List to keep track of the count of each class in the training data.
        online_cache (dict): Cache for storing text and labels for online training.
    """
    def __init__(self, args: ModelArguments, **model_kwargs) -> None:
        """
        Initialize the GenericTransformersModel class.

        Parameters:
            args (ModelArguments): Configuration arguments for the model.
            **model_kwargs: Additional keyword arguments for the model class.
        """
        super().__init__(args)
        self.args = args
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_args.model_name_or_path, use_fast=True)
        self.initialize_model()
        self.args.model_args.max_length = self.tokenizer.model_max_length if not hasattr(self.args.model_args, 'max_length') else self.args.model_args.max_length
        self.online_cache = {"text": [], "llm_label": []}
        print(f"{args.name} model loaded")

    def initialize_model(self) -> None:
        """
        Clean up GPU memory and reload the model.
        """
        torch.cuda.empty_cache()
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model_args.model_name_or_path, num_labels=self.args.num_labels)
        self.model = self.model.to(self.device)

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """
        Train the model using the provided training data.

        Parameters:
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader): DataLoader for validation data.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.model_args.learning_rate)
        best_val_acc = 0

        for epoch in range(self.args.model_args.num_epochs):
            print("Epoch: ", epoch + 1)
            self.model.train()
            pbar = tqdm(train_dataloader)
            for _, batch in enumerate(pbar):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.model_args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to(self.device)
                true_labels = batch[-1].to(self.device)
                outputs = self.model(**encoded_text, labels=true_labels)
                loss = outputs.loss
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        predictions, val_acc = self.evaluate(val_dataloader)
        print("Validation Accuracy: ", val_acc)
        return val_acc

    def train_online(self, data: dict) -> None:
        """
        Train the model using online (incremental) training data.

        Parameters:
            data (dict): Dictionary containing 'text' and 'llm_label' keys for training data.
        """
        online_data = GenericDataset(self.online_cache)
        data = DataLoader(online_data, batch_size=self.args.model_args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.model_args.learning_rate)
        self.model.train()
        
        for _ in range(self.args.model_args.num_epochs):
            for _, batch in enumerate(data):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.model_args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to(self.device)
                true_labels = batch[1].to(self.device)
                outputs = self.model(**encoded_text, labels=true_labels)
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(outputs.logits.view(-1, self.args.num_labels), true_labels.view(-1))
                loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()
        self.cache_clear()

    def inference(self, dataloader: DataLoader) -> torch.tensor:
        """
        Perform inference on the provided data.

        Parameters:
            dataloader (DataLoader): DataLoader for input data.

        Returns:
            torch.tensor: The predicted probabilities for each class.
        """
        self.model.eval()
        probs = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader)):
                encoded_text = self.tokenizer.batch_encode_plus(batch[0], padding='max_length', max_length=self.args.model_args.max_length, truncation=True, return_tensors='pt')
                encoded_text = encoded_text.to(self.device)
                logits = self.model(**encoded_text).logits
                prob = nn.functional.softmax(logits, dim=-1)
                probs = torch.cat((probs, prob))
        return probs

    def evaluate(self, dataloader: DataLoader) -> tuple[torch.tensor, float]:
        """
        Evaluate the model on the provided test data.

        Parameters:
            dataloader (DataLoader): DataLoader for test data.

        Returns:
            tuple[torch.tensor, float]: The predictions and accuracy of the model on the test data.
        """
        raise NotImplementedError("Evaluate method not implemented.")
    
    def predict(self, input: str) -> torch.tensor:
        """
        Predict the probabilities of each class for the given input text.

        Parameters:
            input (str): The input text for prediction.

        Returns:
            torch.tensor: The predicted probabilities for each class.
        """
        self.model.eval()
        with torch.no_grad():
            input = self.tokenizer.encode_plus(input, padding='max_length', max_length=self.args.model_args.max_length, truncation=True, return_tensors='pt')
            input = input.to(self.device)
            logits = self.model(**input).logits
            probs = nn.functional.softmax(logits, dim=-1)
        return probs
