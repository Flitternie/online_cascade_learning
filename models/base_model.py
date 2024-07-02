from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    BaseModel is an abstract base class that defines the structure for any model.
    It serves as a template for creating models and enforces the implementation 
    of key methods required for training, inference, and evaluation.

    Attributes:
    -----------
    args : ModelArguments
        A configuration object containing model arguments and settings.
    online_cache : dict
        A cache to store online training data, including text and labels.
    """
    
    def __init__(self, args):
        """
        Initializes the BaseModel with the given arguments.

        Parameters:
        -----------
        args : ModelArguments
            A configuration object containing model arguments and settings.
        """
        self.args = args
        self.online_cache = {"text": [], "llm_label": []}

    @abstractmethod
    def initialize_model(self) -> None:
        """
        Initializes the model. This method should be implemented 
        by subclasses to perform any necessary model setup or pre-training.
        """
        pass

    def cache_add(self, text: str, label: int) -> None:
        """
        Adds a text and label to the online cache for later training.

        Parameters:
        -----------
        text : str
            The text data to be added to the cache.
        label : int
            The label associated with the text data.
        """
        self.online_cache["text"].append(text)
        self.online_cache["llm_label"].append(label)
    
    def cache_clear(self) -> None:
        """
        Clears the online cache. This method should reset the cache to its 
        initial state.
        """
        self.online_cache = {"text": [], "llm_label": []}

    @abstractmethod
    def train(self, train_data: dict) -> None:
        """
        Trains the model using the provided training data.

        Parameters:
        -----------
        train_data : dict
            A dictionary containing the training data, typically with keys 'text' and 'llm_label'.
        """
        pass

    @abstractmethod
    def train_online(self, train_data: dict) -> None:
        """
        Trains the model online using the provided training data. This method is 
        used for incremental training.

        Parameters:
        -----------
        train_data : dict
            A dictionary containing the training data, typically with keys 'text' and 'llm_label'.
        """
        pass

    @abstractmethod
    def inference(self, data: dict):
        """
        Performs inference on the given data and returns the model predictions.

        Parameters:
        -----------
        data : dict
            A dictionary containing the data to be inferred, typically with a key 'text'.

        Returns:
        --------
        Predictions from the model.
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: dict) -> float:
        """
        Evaluates the model using the provided test data and returns a performance metric.

        Parameters:
        -----------
        test_data : dict
            A dictionary containing the test data, typically with keys 'text' and 'label'.

        Returns:
        --------
        float
            A performance metric, such as accuracy.
        """
        pass

    @abstractmethod
    def predict(self, input: str):
        """
        Makes a prediction for a single input string.

        Parameters:
        -----------
        input : str
            A single input string to be predicted.

        Returns:
        --------
        Prediction for the input string.
        """
        pass

class BaseLLM(ABC):
    """
    BaseLLM is an abstract base class for large language models (LLMs).
    It serves as a template for creating language models, enforcing the implementation
    of essential methods required for making predictions.

    Attributes:
    -----------
    args : ModelArguments
        A configuration object containing model arguments and settings.
    """
    
    def __init__(self, args):
        """
        Initializes the BaseLLM with the given arguments.

        Parameters:
        -----------
        args : ModelArguments
            A configuration object containing model arguments and settings.
        """
        self.args = args

    @abstractmethod
    def predict(self, input: str):
        """
        Makes a prediction for a single input string.

        Parameters:
        -----------
        input : str
            A single input string to be predicted.

        Returns:
        --------
        Prediction for the input string.
        """
        pass