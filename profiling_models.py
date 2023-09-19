import time
import torch
import datasets
import numpy as np
from sklearn.model_selection import train_test_split

from bert import BertModel
from llama import LlamaModel
from online_learning_pipeline import LogisticRegressionModel

from inference_imdb import *
from utils import *

def profiling_lr(lr_config, data):
    model = LogisticRegressionModel(lr_config)
    model.initialize_vectorizer(data['text'])
    model.train(data[:500])
    # warmup
    model.inference(data[500:])
    # profiling and take average
    t = []
    for i in range(5):
        start_time = time.time()
        test_data = data[500:]['text']
        for j in test_data:
            model.predict(j)
        end_time = time.time()
        t.append(end_time - start_time)
    print("LR Profiling Time: ", np.mean(t))

def profiling_bert(bert_config, data):
    model = BertModel(bert_config)
    dataset = GenericDataset(data)
    train_data, val_data = random_split(dataset, [500, 500])
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)
    model.train(train_dataloader, val_dataloader)
    # warmup
    model.inference(val_dataloader)
    # profiling and take average
    t = []
    for i in range(5):
        start_time = time.time()
        test_data = data[500:]['text']
        for j in test_data:
            model.predict(j)
        end_time = time.time()
        t.append(end_time - start_time)
    print("BERT Profiling Time: ", np.mean(t))

def profiling_llama(llama_config, data):
    model = LlamaModel(llama_config)
    dataset = ImdbDataset(data)
    # profiling and take average
    t = []
    for i in range(5):
        start_time = time.time()
        test_data = data[500:]['text']
        for j in test_data:
            model.predict(j)
        end_time = time.time()
        t.append(end_time - start_time)
    print("LLAMA Profiling Time: ", np.mean(t))

def main():
    lr_config = ModelArguments()
    lr_config.confidence_threshold = 1.5
    lr_config.online_batch_size = 8

    bert_config = ModelArguments()
    bert_config.model = "bert-base-uncased"
    bert_config.num_labels = 2
    bert_config.confidence_threshold = 0.63
    bert_config.online_batch_size = 32
    bert_config.online_minibatch_size = 8
    bert_config.epochs = 1

    llama_config = ModelArguments()
    llama_config.model = "meta-llama/Llama-2-7b-chat-hf"
    llama_config.temperature = 0.3
    llama_config.max_new_tokens = 100
    llama_config.top_k = 5

    data = datasets.load_from_disk("data/1000_sampled_imdb")
    data = data.shuffle()
    set_seed(42)
    
    # profiling_lr(lr_config, data)
    # profiling_bert(bert_config, data)
    profiling_llama(llama_config, data)

    

if __name__ == "__main__":
    main()