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
        # print(train_data_vector)
        # print("Training llm labels: ", train_data['llm_label'])
        # print(np.arange(self.args.num_labels))
        self.model.partial_fit(train_data_vector, train_data['llm_label'], classes=np.arange(self.args.num_labels))
        '''
        # REBUTTAL EXPERIMENT: Calculate FLOPs
        # Assuming each feature operation involves 2 FLOPs (1 multiplication, 1 addition)
        # And assuming the gradient computation and weight update involve n FLOPs
        n_features = train_data_vector.shape[1]
        n_samples = train_data_vector.shape[0]
        flops_per_sample = 2 * n_features + n_features  # dot product and weight update
        total_flops = flops_per_sample * n_samples
        with open("./flops.txt", "a") as f:
            f.write(f"LR Training, {total_flops}\n")
        '''
        
    def train_online(self, train_data: dict) -> None:
        self.train(train_data)
        self.cache_clear()

    def inference(self, data: dict):
        test_data = self.vectorizer.transform(data['text'])
        return self.model.predict_proba(test_data)

    def evaluate(self, test_data: dict) -> float:
        predictions = self.inference(test_data)
        correct = 0
        tp, fn = 0, 0
        for i in range(len(predictions)):
            if int(np.argmax(predictions[i])) == test_data['label'][i]:
                correct += 1
            if int(np.argmax(predictions[i])) == 1 and test_data['label'][i] == 1:
                tp += 1
            if int(np.argmax(predictions[i])) == 0 and test_data['label'][i] == 1:
                fn += 1
        recall = tp / (tp + fn)
        print(f"Recall: {recall}")
        return correct / len(predictions)

    def predict(self, input: str):
        test_data = self.vectorizer.transform([input])
        '''
        # REBUTTAL EXPERIMENT: Calculate FLOPs
        # Assuming each multiplication and addition in the dot product counts as 1 FLOP,
        # and there's an additional set of operations for the logistic function,
        # which we'll approximate as equal to the number of features for simplicity.
        n_features = test_data.shape[1]
        flops_for_dot_product = 2 * n_features  # each feature involves a multiplication and an addition
        flops_for_logistic = n_features  # simplified estimate for the logistic function operations
        total_flops = flops_for_dot_product + flops_for_logistic
        with open("./flops.txt", "a") as f:
            f.write(f"LR Inference, {total_flops}\n")
        '''
        return self.model.predict_proba(test_data)

