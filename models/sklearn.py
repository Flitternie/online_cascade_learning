import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import ClassifierMixin
from models.base_model import BaseModel
from utils import ModelArguments

class GenericSklearnModel(BaseModel):
    """
    A generic model class for training and evaluating scikit-learn classifiers.

    Attributes:
        args (ModelArguments): Configuration arguments for the model.
        model (ClassifierMixin): The scikit-learn classifier instance.
        vectorizer (TfidfVectorizer): Vectorizer for transforming text into TF-IDF features.
        class_count (list): List to keep track of the count of each class in the training data.
        online_cache (dict): Cache for storing text and labels for online training.
    """
    def __init__(self, args: ModelArguments, model_class: ClassifierMixin, corpus: list[str], **model_kwargs) -> None:
        """
        Initialize the GenericSklearnModel class.

        Parameters:
            args (ModelArguments): Configuration arguments for the model.
            model_class (ClassifierMixin): The scikit-learn classifier class.
            corpus (list[str]): List of text data for initializing the vectorizer.
            **model_kwargs: Additional keyword arguments for the model class.
        """
        super().__init__(args)
        self.args = args
        self.model = model_class(**model_kwargs)  # Initialize the model with the given class and parameters
        self.vectorizer = TfidfVectorizer()  # Initialize the TF-IDF vectorizer
        self.initialize_model(corpus)  # Fit the vectorizer with the initial data
        self.online_cache = {"text": [], "llm_label": []}
        print(f"{args.name} model initialized.")

    def initialize_model(self, corpus: list[str]) -> None:
        """
        Initialize the TF-IDF vectorizer with the provided corpus.

        Parameters:
            corpus (list[str]): List of text corpus for fitting the vectorizer.
        """
        self.vectorizer.fit(corpus)

    def train(self, train_data: dict) -> None:
        """
        Train the model using the provided training data.

        Parameters:
            train_data (dict): Dictionary containing 'text' and 'llm_label' keys for training data.
        """
        # Transform text data into TF-IDF vectors
        train_data_vector = self.vectorizer.transform(train_data['text'])
        if "partial_fit" in dir(self.model):
            # Train the model using partial fit
            self.model.partial_fit(train_data_vector, train_data['llm_label'], classes=np.arange(self.args.num_labels))
        else:
            # Train the model using fit
            self.model.fit(train_data_vector, train_data['llm_label'])

    def train_online(self, train_data: dict) -> None:
        """
        Train the model using online (incremental) training data.

        Parameters:
            train_data (dict): Dictionary containing 'text' and 'llm_label' keys for training data.
        """
        # Transform text data into TF-IDF vectors
        train_data_vector = self.vectorizer.transform(train_data['text'])
        if "partial_fit" in dir(self.model):
            # Train the model using partial_fit and clean the cache
            self.model.partial_fit(train_data_vector, train_data['llm_label'], classes=np.arange(self.args.num_labels))
            self.cache_clear()
        else:
            # Train the model using fit without cache clearing
            self.model.fit(train_data_vector, train_data['llm_label'])

    def inference(self, data: dict):
        """
        Perform inference on the provided data.

        Parameters:
            data (dict): Dictionary containing 'text' key for input data.

        Returns:
            numpy.ndarray: The predicted probabilities for each class.
        """
        test_data = self.vectorizer.transform(data['text'])
        return self.model.predict_proba(test_data)

    def evaluate(self, test_data: dict) -> float:
        """
        Evaluate the model on the provided test data.

        Parameters:
            test_data (dict): Dictionary containing 'text' and 'label' keys for test data.

        Returns:
            float: The accuracy of the model on the test data.
        """
        raise NotImplementedError("Evaluate method not implemented.")

    def predict(self, input: str):
        """
        Predict the probabilities of each class for the given input text.

        Parameters:
            input (str): The input text for prediction.

        Returns:
            numpy.ndarray: The predicted probabilities for each class.
        """
        test_data = self.vectorizer.transform([input])
        return self.model.predict_proba(test_data)