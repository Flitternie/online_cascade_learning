import importlib
import os

from .wrapper import ModelWrapper
from .sklearn import GenericSklearnModel
from .transformers import GenericTransformersModel
from .openai import GenericOpenAIModel

def llm_factory(llm_config, data_config, **llm_kwargs):
    # llm_kwargs = llm_kwargs | llm_config.model_args.serialize() # Merge the two dictionaries
    data_module = importlib.import_module(data_config.env)
    if os.path.isfile(llm_config.source): # If the LLM source is a file
        llm_labels = open(llm_config.source, "r").readlines()
        llm = [int(data_module.postprocess(l.strip())) for l in llm_labels]
    elif "openai" in llm_config.source.lower():
        llm = GenericOpenAIModel(llm_config, data_module, **llm_kwargs)
    else:
        raise NotImplementedError(f"LLM source {llm_config.source} not supported.")
    return llm

def model_factory(model_config, data_config, **model_kwargs):
    model_config.num_labels = data_config.num_labels
    model_kwargs = model_kwargs | model_config.model_args.serialize() # Merge the two dictionaries

    if "sklearn" in model_config.source.lower():
        try:
            # Dynamically import the module and class
            module = importlib.import_module(model_config.source)
            model_class = getattr(module, model_config.model)
        except:
            raise ImportError(f"Model {model_config.model} not found in {model_config.source}")
        model = GenericSklearnModel(model_config, model_class, corpus=data_config.corpus, **model_kwargs)
    elif "transformers" in model_config.source.lower():
        assert model_config.model == "AutoModelForSequenceClassification", "Only AutoModelForSequenceClassification supported for transformers."
        model = GenericTransformersModel(model_config, **model_kwargs)
    else:
        raise NotImplementedError(f"Model source {model_config.source} not supported.")
    return model
