import datasets
import pandas as pd
import numpy as np
import argparse
import importlib

import models.lr as lr
import models.bert as bert

from utils import *
from cascade.online import *
        
def main(mu):
    print("cost coefficient: ", mu)
    data_env = 'data.imdb'
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

    # REBUTTAL EXPERIMENT
    # sort the order of data by the length of text by first converting to pandas dataframe
    data = pd.DataFrame(data)
    data = data.sort_values(by='text', key=lambda x: x.str.len())
    data = datasets.Dataset.from_pandas(data)
    



    wrappers = []

    lr_config = ModelArguments()
    lr_config.num_labels = 2
    lr_config.cache_size = 8
    lr_config.cost = 1 # 110M for bert-base
    lr_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr_model = lr.LogisticRegressionModelSkLearn(lr_config, data=data['text'])
    
    lr_wrapper = ModelWrapper(lr_model, lr_model.args)
    lr_wrapper.name = "LR"
    lr_wrapper.learning_rate = 0.0007
    lr_wrapper.regularization = 0.0001
    lr_wrapper.decaying_factor = 0.97
    lr_wrapper.calibration = 0.4
    lr_wrapper.to(lr_wrapper.device)
    wrappers.append(lr_wrapper)
    
    bert_base_config = ModelArguments()
    bert_base_config.num_labels = 2
    bert_base_config.model = "bert-base-uncased"
    bert_base_config.cache_size = 16
    bert_base_config.batch_size = 8
    bert_base_config.num_epochs = 5
    bert_base_config.cost = 1182 # 130B for GPT-3
    bert_base_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_base_model = bert.BertModel(bert_base_config)
    
    bert_base_wrapper = ModelWrapper(bert_base_model, bert_base_model.args)
    bert_base_wrapper.name = "BERT-base"
    bert_base_wrapper.learning_rate = 0.0007
    bert_base_wrapper.regularization = 0.0001
    bert_base_wrapper.decaying_factor = 0.95
    bert_base_wrapper.calibration = 0.3
    bert_base_wrapper.to(bert_base_wrapper.device) 
    wrappers.append(bert_base_wrapper)

    pipeline(data_module, data, wrappers, mu, log_dir="./logs/rebuttal/ds_length/gpt/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.02)
    # for mu in np.arange(0.00001, 0.0001, 0.00002):
    #     main(mu)
    for mu in np.arange(0.00011, 0.0002, 0.00002):
        main(mu)
    # for mu in np.arange(0.0002, 0.001, 0.0001):
    #     main(mu)