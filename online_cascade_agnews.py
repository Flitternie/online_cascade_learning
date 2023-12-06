import datasets
import numpy as np
import argparse

import lr
import bert

from utils import *
from cascades.online_pipeline import *
        
def main(mu):
    print("cost coefficient: ", mu)
    data_env = 'inference_agnews'
    data_module = importlib.import_module(data_env)
    
    set_seed(42)
    data = datasets.load_from_disk("./data/10000_sampled_agnews")

    llama_labels = open("./llama_results/10000_sampled_agnews_llama.txt", "r").readlines()
    llama_labels = [int(l.strip()) for l in llama_labels]
    total, correct = 0, 0
    for i, d in enumerate(data):
        if data[i]['label'] == llama_labels[i]:
            correct += 1
        total += 1
    print(f"LLAMA Accuracy: {correct/total}")
    
    def update_labels(example, idx):
        example['llm_label'] = llama_labels[idx]
        return example
    
    data = data.map(update_labels, with_indices=True)
    total, correct = 0, 0
    for i, d in enumerate(data):
        if data[i]['llm_label'] == llama_labels[i]:
            correct += 1
        total += 1
    print(f"LLAMA Accuracy: {correct/total}")

    # split data into train and test
    data = data.train_test_split(test_size=0.5)

    data = data['test'].shuffle()

    lr_config = ModelArguments()
    lr_config.num_labels = 4
    lr_config.cache_size = 8
    lr_config.cost = 1 #110M for bert-base
    lr_model = lr.LogisticRegressionModelSkLearn(lr_config, data=data['text'])
    lr_wrapper = ModelWrapper(lr_model, lr_model.args)
    
    lr_wrapper.learning_rate = 0.0007
    lr_wrapper.regularization = 0.0001
    lr_wrapper.decaying_factor = 0.97
    lr_wrapper.calibration = 0.1
    lr_wrapper.to('cuda')
    
    bert_config = ModelArguments()
    bert_config.num_labels = 4
    bert_config.model = "bert-base-uncased"
    bert_config.cache_size = 16
    bert_config.batch_size = 8
    bert_config.num_epochs = 5
    bert_config.cost = 63 #7B for llama2-7b
    bert_model = bert.BertModel(bert_config)
    
    bert_wrapper = ModelWrapper(bert_model, bert_model.args)
    bert_wrapper.learning_rate = 0.0007
    bert_wrapper.regularization = 0.0001
    bert_wrapper.decaying_factor = 0.95
    bert_wrapper.calibration = 0.1
    bert_wrapper.to('cuda')

    # llama_config = ModelArguments()
    # llama_config.model = "meta-llama/Llama-2-7b-chat-hf"
    # llama_config.temperature = 0.3
    # llama_config.max_new_tokens = 100
    # llama_config.top_k = 5
    # llama_model = llama.LlamaModel(llama_config)

    pipeline(data_module, data, lr_wrapper, bert_wrapper, mu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.02)
    for mu in np.arange(0.0015, 0.02, 0.001):
        main(mu)
