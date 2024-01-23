from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from utils import *  

class LogisticRegressionModelSkLearn():
    def __init__(self, args: ModelArguments, data: list):
        self.args = args
        self.model = SGDClassifier(loss='log_loss')
        self.vectorizer = TfidfVectorizer()
        self.initialize_vectorizer(data)
        if "class_weight" in dir(self.args):
            self.class_weight = self.args.class_weight
            self.class_count = [0 for _ in range(self.args.num_labels)]
        self.online_cache = {"text": [], "llm_label": []}
        print("Logistic Regression Model initialized")
    
    def initialize_vectorizer(self, data: list[str]) -> None:
        self.vectorizer.fit(data)
    
    def cache_add(self, text: str, label: int) -> None:
        self.online_cache["text"].append(text)
        self.online_cache["llm_label"].append(label)
    
    def cache_clear(self) -> None:
        self.online_cache = {"text": [], "llm_label": []}

    def train(self, train_data: dict) -> None:
        if "class_weight" in dir(self.args):
            for label in train_data['llm_label']:
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
    
    def train_online(self, train_data: dict) -> None:
        self.train(train_data)

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

