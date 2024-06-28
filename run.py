import numpy as np
import pandas as pd
import argparse
import datasets
import importlib
import torch

from models import model_factory, ModelWrapper
from utils import set_seed, load_config
from cascade.online import pipeline
        
def main(mu, config):
    data_env = config.data.env
    data_module = importlib.import_module(data_env)
    
    set_seed(config.seed)
    data = datasets.Dataset.from_pandas(pd.read_csv(config.data.path))
    data = data_module.preprocess(data)

    llm_labels = open(config.llm.source, "r").readlines()
    llm_labels = [int(data_module.postprocess(l.strip())) for l in llm_labels]
    total, correct = 0, 0
    for i, _ in enumerate(data):
        if data[i]['label'] == llm_labels[i]:
            correct += 1
        total += 1
    print(f"LLM Accuracy: {correct/total}")
    
    def update_labels(example, idx):
        example['llm_label'] = llm_labels[idx]
        return example
    
    data = data.map(update_labels, with_indices=True)
    total, correct = 0, 0
    for i, _ in enumerate(data):
        if data[i]['llm_label'] == llm_labels[i]:
            correct += 1
        total += 1
    assert correct/total == 1.0 # should be 1.0

    # split data into train and test
    data = data.shuffle()
    data = data.train_test_split(test_size=0.5)
    data = data['test']
    config.data.corpus = data['text']

    wrappers = []

    for model_config in config.models:
        model_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model_factory(model_config, data_config=config.data)

        model_wrapper = ModelWrapper(model, model_config)
        model_wrapper.to(model_wrapper.device)
        wrappers.append(model_wrapper)
            
    pipeline(data_module, data, wrappers, mu, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    
    for mu_range in config.mu:
        if isinstance(mu_range, int):
            print("cost coefficient: ", mu_range)
            main(mu_range, config)
        elif isinstance(mu_range, list) and len(mu_range) == 3:
            for mu in np.arange(*mu_range):
                print("cost coefficient: ", mu)
                main(mu, config)
        else:
            raise ValueError("Invalid mu range, should be an int or a list of 3 elements")