import datasets
import pandas as pd
import numpy as np
import argparse
import importlib
import sys
import os

# Get the absolute path of the project folder
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import models.lr as lr
import models.bert as bert

from utils import *
from cascade.online import *
        
def main(mu):
    print("cost coefficient: ", mu)
    data_env = 'data.isear'
    data_module = importlib.import_module(data_env)
    
    set_seed(42)
    data = datasets.Dataset.from_pandas(pd.read_csv("./data/isear_preprocessed.csv"))

    isear_to_id = data_module.isear_to_id

    # Change labels to id
    data = data.map(lambda e: {'label': isear_to_id[e['label']]})

    llm_labels = open("./gpt_results/gpt3.5/isear_gpt3.5_turbo_1106.txt", "r").readlines()
    llm_labels = [int(data_module.postprocess(l.strip())) for l in llm_labels]
    total, correct = 0, 0
    for i, d in enumerate(data):
        if data[i]['label'] == llm_labels[i]:
            correct += 1
        total += 1
    print(f"LLM Accuracy: {correct/total}")
    
    def update_labels(example, idx):
        example['llm_label'] = llm_labels[idx]
        return example
    
    data = data.map(update_labels, with_indices=True)
    total, correct = 0, 0
    for i, d in enumerate(data):
        if data[i]['llm_label'] == llm_labels[i]:
            correct += 1
        total += 1
    assert correct/total == 1.0 # should be 1.0

    # split data into train and test
    data = data.shuffle()
    data = data.train_test_split(test_size=0.5)
    data = data['test']

    wrappers = []

    lr_config = ModelArguments()
    lr_config.num_labels = 7
    lr_config.cache_size = 8
    lr_config.cost = 1 # 110M for bert-base
    lr_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr_model = lr.LogisticRegressionModelSkLearn(lr_config, data=data['text'])
    
    lr_wrapper = ModelWrapper(lr_model, lr_model.args)
    lr_wrapper.name = "LR"
    lr_wrapper.learning_rate = 0.0007
    lr_wrapper.regularization = 0.0001
    lr_wrapper.decaying_factor = 0.8
    lr_wrapper.calibration = 0.15
    lr_wrapper.to(lr_wrapper.device)
    wrappers.append(lr_wrapper)
    
    bert_base_config = ModelArguments()
    bert_base_config.num_labels = 7
    bert_base_config.model = "bert-base-uncased"
    bert_base_config.cache_size = 16
    bert_base_config.batch_size = 8
    bert_base_config.num_epochs = 5
    bert_base_config.cost = 1182  # 340M for bert-large
    bert_base_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_base_model = bert.BertModel(bert_base_config)
    
    bert_base_wrapper = ModelWrapper(bert_base_model, bert_base_model.args)
    bert_base_wrapper.name = "BERT-base"
    bert_base_wrapper.learning_rate = 0.0007
    bert_base_wrapper.regularization = 0.0001
    bert_base_wrapper.decaying_factor = 0.9
    bert_base_wrapper.calibration = 0.45
    bert_base_wrapper.to(bert_base_wrapper.device) 
    wrappers.append(bert_base_wrapper)

    pipeline(data_module, data, wrappers, mu, log_dir="./logs_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.02)
    # for mu in np.arange(0.005, 0.006, 0.0001):
    #     main(mu)
    main(0.0012)
