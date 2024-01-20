import datasets
import pandas as pd
import numpy as np
import argparse
import importlib

import models.lr as lr
import models.bert as bert

from utils import *
from cascades.online_pipeline_general import *
        
def main(mu):
    print("cost coefficient: ", mu)
    data_env = 'inference_imdb'
    data_module = importlib.import_module(data_env)
    
    set_seed(42)
    data = datasets.Dataset.from_pandas(pd.read_csv("./data/imdb_preprocessed.csv"))

    llm_labels = open("./gpt_results/gpt3.5/imdb_gpt3.5_turbo_1106.txt", "r").readlines()
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
    lr_config.num_labels = 2
    lr_config.cache_size = 8
    lr_config.cost = 1 #110M for bert-base
    lr_model = lr.LogisticRegressionModelSkLearn(lr_config, data=data['text'])
    
    lr_wrapper = ModelWrapper(lr_model, lr_model.args)
    lr_wrapper.name = "LR"
    lr_wrapper.learning_rate = 0.0007
    lr_wrapper.regularization = 0.0001
    lr_wrapper.decaying_factor = 0.97
    lr_wrapper.calibration = 0.4
    lr_wrapper.to('cuda')
    wrappers.append(lr_wrapper)
    
    bert_base_config = ModelArguments()
    bert_base_config.num_labels = 2
    bert_base_config.model = "bert-base-uncased"
    bert_base_config.cache_size = 16
    bert_base_config.batch_size = 8
    bert_base_config.num_epochs = 5
    bert_base_config.cost = 1182 #130B for GPT-3
    bert_base_model = bert.BertModel(bert_base_config)
    
    bert_base_wrapper = ModelWrapper(bert_base_model, bert_base_model.args)
    bert_base_wrapper.name = "BERT-base"
    bert_base_wrapper.learning_rate = 0.0007
    bert_base_wrapper.regularization = 0.0001
    bert_base_wrapper.decaying_factor = 0.95
    bert_base_wrapper.calibration = 0.3
    bert_base_wrapper.to('cuda')
    wrappers.append(bert_base_wrapper)

    pipeline(data_module, data, wrappers, mu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.02)
    for mu in np.arange(0.0001, 0.001, 0.0001):
        main(mu)
