import os
from typing import List, Dict, Any
import litellm
from litellm import completion
from models.base_model import BaseLLM

litellm.num_retries_per_request = 3

class GenericLLM(BaseLLM):
    """
    A class to interact with LiteLLM interface for natural language processing tasks.

    Attributes:
        model (str): The name of the LLM to use.
        model_config (ModelArguments): The configuration object containing the model details.
        supported_params (List[str]): A list of supported parameters for the LLM API.
        data_module (DataModule): An instance containing the system prompt, user prompt, and label list.
        system_prompt (str): The system prompt for the LLM.
        user_prompt (str): The user prompt for the LLM.
        label_list (List[str]): A list of the legit labels for the classification task.
        kwargs (dict): Additional keyword arguments for the LLM API.
        logit_bias (dict): A dictionary mapping token IDs to logit biases.
    """
    def __init__(self, model_config: Any, data_module: Any, **kwargs: Any) -> None:
        """
        Initialize the GenericLLM with the provided model configuration and data module.

        Args:
            model_config (Any): The configuration object containing the model details.
            data_module (Any): The data module object containing prompts and label list.
            **kwargs (Any): Additional keyword arguments for the OpenAI API.
        """
        self.model = model_config.model.lower()
        self.model_config = model_config
        assert self.model_config.source.lower() in litellm.provider_list, f"Model source {self.model_config.source} not supported."
        assert self.model in litellm.models_by_provider[self.model_config.source.lower()], f"Model {self.model} not supported."
        self._set_env()
        self.supported_params = litellm.get_supported_openai_params(self.model)
        
        self.data_module = data_module
        self.system_prompt = data_module.SystemPrompt
        self.user_prompt = data_module.UserPrompt
        self.label_list = data_module.LabelList
        self.kwargs = kwargs
        self._set_logit_bias(self.label_list)

    def _set_env(self) -> None:
        """
        Set the environment variables for the model.
        """
        env_var = f"{self.model_config.source.upper()}_API_KEY"
        if not os.getenv(env_var):
            try:
                os.environ[env_var] = self.model_config.api_key
            except:
                raise ValueError(f"API key not found in environment variable {env_var} or model configuration.")
    
    def _set_logit_bias(self, label_list: List[str]) -> None:
        """
        Set the logit bias for the provided label list.

        Args:
            label_list (List[str]): A list of labels for which to set logit biases.
        """
        self.label_ids = []
        for label_name in label_list:
            label_id = litellm.encode(model=self.model, text=label_name)
            if isinstance(label_id, list):
                self.label_ids += label_id
            else:
                self.label_ids.append(label_id)
        self.logit_bias = {label_id: 100 for label_id in self.label_ids}
        self.kwargs["logit_bias"] = self.logit_bias

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
        Predict the label for a given query using the model.

        Args:
            query (str): The input query for which to predict the label.

        Returns:
            int: The predicted label.
        """
        # filter out the model configurations that are not supported by the API
        self.kwargs = {key: value for key, value in self.kwargs.items() if key in self.supported_params}
        response = completion(
            model=self.model, 
            messages=[
                {"role": "system", "content": self.system_prompt}, 
                {"role": "user", "content": self.user_prompt.format(query)}
            ],
            **self.kwargs
            )
        text_output = response.choices[0].message.content
        prediction = self.data_module.postprocess(text_output)
        
        return prediction
