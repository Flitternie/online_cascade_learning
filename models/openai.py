import os
from typing import List, Dict, Any
import tiktoken
from openai import OpenAI
from models.base_model import BaseLLM

class GenericOpenAIModel(BaseLLM):
    """
    A class to interact with OpenAI models for natural language processing tasks.

    Attributes:
        model (str): The name of the OpenAI model to use.
        data_module (DataModule): An instance containing the system prompt, user prompt, and label list.
        kwargs (dict): Additional keyword arguments for the OpenAI API.
        api_key (str): The API key for accessing OpenAI.
        org_id (str): The organization ID for accessing OpenAI.
        client (OpenAI): An instance of the OpenAI client.
        logit_bias (dict): A dictionary mapping token IDs to logit biases.
    """

    def __init__(self, model_config: Any, data_module: Any, **kwargs: Any) -> None:
        """
        Initialize the GenericOpenAIModel with the provided model configuration and data module.

        Args:
            model_config (Any): The configuration object containing the model details.
            data_module (Any): The data module object containing prompts and label list.
            **kwargs (Any): Additional keyword arguments for the OpenAI API.
        """
        self.model = model_config.model
        self.data_module = data_module
        self.system_prompt = data_module.SystemPrompt
        self.user_prompt = data_module.UserPrompt
        self.label_list = data_module.LabelList
        self.kwargs = kwargs
        self._set_logit_bias(self.label_list)

        self.api_key = os.getenv("OPENAI_KEY") or self._get_file_var("~/OPENAI_KEY")
        self.org_id = os.getenv("OPENAI_ORG") or self._get_file_var("~/OPENAI_ORG")

        if not self.api_key or not self.org_id:
            raise ValueError("OpenAI API key and organization ID not found.")

        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.org_id,
        )
    
    def _get_file_var(self, file_path: str) -> str:
        """
        Retrieve the value of a variable stored in a file.

        Args:
            file_path (str): The path to the file containing the variable.

        Returns:
            str: The value of the variable, or None if the file is not found or an error occurs.
        """
        try:
            with open(os.path.expanduser(file_path), 'r') as file:
                return file.read().strip()
        except (FileNotFoundError, IOError):
            return None
    
    def _set_logit_bias(self, label_list: List[str]) -> None:
        """
        Set the logit bias for the provided label list.

        Args:
            label_list (List[str]): A list of labels for which to set logit biases.
        """
        encoder = tiktoken.encoding_for_model(self.model)
        self.label_ids = []
        for label_name in label_list:
            label_id = encoder.encode(label_name)
            if isinstance(label_id, list):
                self.label_ids += label_id
            else:
                self.label_ids.append(label_id)
        self.logit_bias = {label_id: 100 for label_id in self.label_ids}

    def log_probs_to_dict(self, logprobs: List[Any]) -> Dict[str, float]:
        """
        Convert log probabilities to a dictionary.

        Args:
            logprobs (List[Any]): A list of log probabilities.

        Returns:
            Dict[str, float]: A dictionary mapping tokens to their log probabilities.
        """
        token_dict = {}
        for prob in logprobs:
            token_dict[prob.token] = prob.logprob
        return token_dict

    def predict(self, query: str) -> int:
        """
        Predict the label for a given query using the OpenAI model.

        Args:
            query (str): The input query for which to predict the label.

        Returns:
            int: The predicted label.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            seed=42,
            logprobs=True,
            top_logprobs=5,
            logit_bias=self.logit_bias,
            temperature=0.3,
            max_tokens=5,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt.format(query)},
            ],
            **self.kwargs
        )
        text_output = response.choices[0].message.content
        prediction = self.data_module.postprocess(text_output)
        log_probs = self.log_probs_to_dict(response.choices[0].logprobs.content[0].top_logprobs)
        
        return prediction
