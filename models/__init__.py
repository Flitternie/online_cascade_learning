from .wrapper import ModelWrapper
from .sklearn import GenericSklearnModel
from .transformers import GenericTransformersModel
import importlib

def model_factory(model_config, data_config, **model_kwargs):
    model_config.num_labels = data_config.num_labels
    model_kwargs = model_kwargs | model_config.model_args.serialize() # merge the two dictionaries

    if 'sklearn' in model_config.source:
        try:
            # Dynamically import the module and class
            module = importlib.import_module(model_config.source)
            model_class = getattr(module, model_config.model)
        except:
            raise ImportError(f"Model {model_config.model} not found in {model_config.source}")
        model = GenericSklearnModel(model_config, model_class, corpus=data_config.corpus, **model_kwargs)
    elif 'transformers' in model_config.source:
        assert model_config.model == "AutoModelForSequenceClassification", "Only AutoModelForSequenceClassification supported for transformers."
        model = GenericTransformersModel(model_config, **model_kwargs)
    else:
        raise NotImplementedError(f"Model source {model_config.source} not supported.")
    return model
