from datasets import Dataset
from typing import Any, Union, Dict, List

class CustomDataset:
    """
    A custom dataset class that integrates LLM labels into the dataset.

    Attributes:
        _data (Dataset): The dataset object.
        _llm_model (Any): The large language model object used for prediction.
    """

    def __init__(self, data: Dataset, llm_model: Any) -> None:
        """
        Initialize the CustomDataset with data and an LLM model.

        Args:
            data (Dataset): The dataset object.
            llm_model (Any): The large language model object.
        """
        self._data = data
        self._llm_model = llm_model

    @classmethod
    def create(cls, data: Dataset, llm: Union[List[str], Any]) -> Union[Dataset, 'CustomDataset']:
        """
        Create a CustomDataset instance or map the dataset to include LLM labels.

        Args:
            data (Dataset): The dataset object.
            llm (Union[List[str], Any]): A list of LLM labels or an LLM model object.

        Returns:
            Union[Dataset, CustomDataset]: The mapped dataset with LLM labels or a CustomDataset instance.
        """
        if isinstance(llm, list):
            # Map the dataset to include the LLM labels from the list
            def add_llm_label(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
                example['llm_label'] = llm[idx]
                return example

            mapped_data = data.map(add_llm_label, with_indices=True)
            return mapped_data
        else:
            return cls(data, llm_model=llm)

    def __getitem__(self, key: Union[str, int]) -> Any:
        """
        Get an item from the dataset by key.

        Args:
            key (Union[str, int]): The key to retrieve the item.

        Returns:
            Any: The retrieved item.
        """
        if isinstance(key, str):
            return self._data[key]
        elif isinstance(key, int):
            item = self._data[key]
            item['llm_label'] = self._get_llm_label(item['text'])
            return item
        else:
            raise TypeError("Invalid key type. Must be int or str.")

    def _get_llm_label(self, text: str) -> Any:
        """
        Get the LLM label for a given text.

        Args:
            text (str): The input text.

        Returns:
            Any: The predicted LLM label.
        """
        return self._llm_model.predict(text)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._data)

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute from the dataset.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute value.
        """
        return getattr(self._data, name)

    def shuffle(self, seed: int = None) -> 'CustomDataset':
        """
        Shuffle the dataset.

        Args:
            seed (int, optional): The seed for shuffling. Defaults to None.

        Returns:
            CustomDataset: The shuffled dataset.
        """
        shuffled_data = self._data.shuffle(seed=seed)
        return CustomDataset(shuffled_data, llm_model=self._llm_model)

    def train_test_split(self, test_size: float = 0.5) -> Dict[str, 'CustomDataset']:
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.5.

        Returns:
            Dict[str, CustomDataset]: A dictionary containing the training and testing datasets.
        """
        split_data = self._data.train_test_split(test_size=test_size)
        return {
            'train': CustomDataset(split_data['train'], llm_model=self._llm_model),
            'test': CustomDataset(split_data['test'], llm_model=self._llm_model)
        }


class GenericDataset(Dataset):
    """
    A generic dataset class to handle data loading and processing.

    Attributes:
        data (dict): The data dictionary.
        text (List[str]): The list of text data.
        labels (List[Any]): The list of labels.
        llm_labels (List[Any], optional): The list of LLM labels.
        num_labels (int): The number of unique labels.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Initialize the GenericDataset with data.

        Args:
            data (Dict[str, Any]): The data dictionary.
        """
        self.data = data
        self.text = data['text']
        try:
            self.labels = data['label']
        except KeyError:
            self.labels = data['llm_label']
        try:
            self.llm_labels = data['llm_label']
        except KeyError:
            pass
        self.num_labels = len(set(self.labels))
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.text)
    
    def __getitem__(self, idx: int) -> Union[tuple[str, int], tuple[str, int, int]]:
        """
        Get an item from the dataset by index.

        Args:
            idx (int): The index of the item.

        Returns:
            Union[tuple[str, int], tuple[str, int, int]]: The text, label, and optionally the LLM label.
        """
        try:
            return self.text[idx], self.labels[idx], self.llm_labels[idx]
        except KeyError:
            return self.text[idx], self.labels[idx]
