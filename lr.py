import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

from utils import *  

class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return nn.functional.softmax(out, dim=-1)

class LogisticRegressionModel():
    def __init__(self, args: ModelArguments, data: list):
        self.args = args
        self.vectorizer = TfidfVectorizer()
        self.dimentionality_reduction = PCA(n_components=512)
        self.args.input_dim = self.initialize_vectorizer(data)
        self.model = LogisticRegression(self.args.input_dim, self.args.num_labels)
        self.model = self.model.to('cuda')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.online_cache = {"text": [], "label": []}
        if "class_weight" in dir(self.args):
            self.class_weight = self.args.class_weight
            self.class_count = [0 for _ in range(self.args.num_labels)]
        print("Logistic Regression Model initialized")
        
    def initialize_vectorizer(self, data: list[str]) -> int:
        data = self.vectorizer.fit_transform(data).toarray()
        data = self.dimentionality_reduction.fit_transform(data)
        return data.shape[-1]

    def cache_add(self, text: str, label: int) -> None:
        self.online_cache["text"].append(text)
        self.online_cache["label"].append(label)
    
    def cache_clear(self) -> None:
        self.online_cache = {"text": [], "label": []}

    def train(self, train_data: dict) -> None:
        self.model.train()
        train_data_vector = self.vectorizer.transform(train_data['text']).toarray()
        train_data_vector = self.dimentionality_reduction.transform(train_data_vector)
        train_data_vector = torch.tensor(train_data_vector, dtype=torch.float32).to('cuda')
        for epoch in range(self.args.num_epochs):
            labels = torch.tensor(train_data['label']).to('cuda')
            logits = self.model(train_data_vector)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels)
            loss.mean().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
    
    def inference(self, data: list[str]) -> list[int]:
        self.model.eval()
        data = self.vectorizer.transform(data).toarray()
        data = self.dimentionality_reduction.transform(data)
        data = torch.tensor(data, dtype=torch.float32).to('cuda')
        output = self.model(data).to('cpu')
        return torch.argmax(output, dim=-1).tolist()
    
    def evaluate(self, test_data: dict) -> float:
        predictions = self.inference(test_data['text'])
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == test_data['label'][i]:
                correct += 1
        return correct / len(predictions)

    def predict(self, input: str) -> tuple[torch.Tensor, float]:
        self.model.eval()
        with torch.no_grad():
            test_data = self.dimentionality_reduction.transform(self.vectorizer.transform([input]).toarray())
            test_data = torch.tensor(test_data, dtype=torch.float32).to('cuda')
            output = self.model(test_data).to('cpu')
        return output


class LogisticRegressionModelSkLearn():
    def __init__(self, args: ModelArguments, data: list):
        self.args = args
        self.model = SGDClassifier(loss='log_loss')
        self.vectorizer = TfidfVectorizer()
        self.initialize_vectorizer(data)
        if "class_weight" in dir(self.args):
            self.class_weight = self.args.class_weight
            self.class_count = [0 for _ in range(self.args.num_labels)]
        self.online_cache = {"text": [], "label": []}
        print("Logistic Regression Model initialized")
    
    def initialize_vectorizer(self, data: list[str]) -> None:
        self.vectorizer.fit(data)
    
    def cache_add(self, text: str, label: int) -> None:
        self.online_cache["text"].append(text)
        self.online_cache["label"].append(label)
    
    def cache_clear(self) -> None:
        self.online_cache = {"text": [], "label": []}

    def train(self, train_data: dict) -> None:
        if "class_weight" in dir(self.args):
            for label in train_data['label']:
                self.class_count[int(label)] += 1
            if self.args.class_weight == "balanced":
                # balanced class weights computed by: n_samples / (n_classes * np.bincount(y))
                self.class_weight =  {i: sum(self.class_count) / max(( self.args.num_labels * self.class_count[i] ), 1) for i in range(self.args.num_labels)}
                # normalize to sum up to 1
                self.class_weight = {k: v / sum(self.class_weight.values()) for k, v in self.class_weight.items()}    
            assert isinstance(self.class_weight, dict), f"Class weight is not a dict: {self.class_weight}"
            assert len(self.class_weight.keys()) == self.args.num_labels, f"Class weight keys: {self.class_weight.keys()} not equal to num_labels: {self.args.num_labels}"
            assert equal(sum(self.class_weight.values()), 1), f"Sum of class weights is {sum(self.class_weight.values())}"
            self.model.class_weight = sort_dict_by_key(self.class_weight)
        train_data_vector = self.vectorizer.transform(train_data['text'])
        self.model.partial_fit(train_data_vector, train_data['llm_label'], classes=np.arange(self.args.num_labels))
    
    def inference(self, data: dict):
        test_data = self.vectorizer.transform(data['text'])
        return self.model.predict_proba(test_data)

    def evaluate(self, test_data: dict) -> float:
        predictions = self.inference(test_data)
        correct = 0
        for i in range(len(predictions)):
            if int(np.argmax(predictions[i])) == test_data['label'][i]:
                correct += 1
        return correct / len(predictions)

    def predict(self, input: str):
        test_data = self.vectorizer.transform([input])
        return self.model.predict_proba(test_data)

