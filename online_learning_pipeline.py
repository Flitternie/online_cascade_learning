import os
# set CUDA_VISIBLE_DEVICES before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

import bert
import llama

from inference_imdb import *
from utils import *

class ModelArguments():
    def __init__(self) -> None:
        pass

class LogisticRegressionModel():
    def __init__(self, args):
        self.args = args
        self.model = SGDClassifier(loss='log_loss')
        self.vectorizer = TfidfVectorizer()
        print("Logistic Regression Model initialized")
    
    def initialize_vectorizer(self, data):
        self.vectorizer.fit_transform(data)

    def train(self, train_data):
        # train_data = self.vectorizer.fit_transform(train_data['text'])
        train_data_vector = self.vectorizer.transform(train_data['text'])
        self.model.partial_fit(train_data_vector, train_data['label'], classes=np.array([0,1]))

    def generate(self, input):
        test_data = self.vectorizer.transform([input])
        return self.model.predict(test_data), self.model.decision_function(test_data)


lr_config = ModelArguments()
lr_config.warmup = 20
lr_config.confidence_threshold = 1.5
lr_config.online_batch_size = 8

bert_config = ModelArguments()
bert_config.model = "bert-base-uncased"
bert_config.num_labels = 2
bert_config.confidence_threshold = 0.63
bert_config.online_batch_size = 32
bert_config.online_minibatch_size = 8

llama_config = ModelArguments()
llama_config.model = "meta-llama/Llama-2-7b-chat-hf"
llama_config.temperature = 0.3
llama_config.max_new_tokens = 100
llama_config.top_k = 5

set_seed(42)

data = datasets.load_from_disk("data/1000_sampled_imdb")
data = data.shuffle()


lr_model = LogisticRegressionModel(lr_config)
lr_model.initialize_vectorizer(data['text'])
bert_model = bert.BertModel(bert_config)
llama_model = llama.LlamaModel(llama_config)

online_data_cache_for_lr = {
    "text": [],
    "label": []
}
online_data_cache_for_bert = {
    "text": [],
    "label": []
}

f = open("pipeline_details.txt", "w+")

overall_lr_correct, overall_bert_correct, overall_llama_correct = 0, 0, 0
num_lr, num_bert, num_llama = 0, 0, 0
by_lr, by_bert = 0, 0
pipeline_correct = 0
update_lr, update_bert = 0, 0

for i, item in enumerate(data):
    # go to smaller model first
    if len(online_data_cache_for_lr['label']) == lr_config.online_batch_size:
        lr_model.train(online_data_cache_for_lr)
        online_data_cache_for_lr["text"] = []
        online_data_cache_for_lr["label"] = []
        # print("LR Updated")
        update_lr += 1
    if i < lr_config.warmup:
        lr_confidence = 0
    else:
        lr_output, lr_confidence = lr_model.generate(item['text'])
        lr_output, lr_confidence = lr_output[0], lr_confidence[0]
        if int(lr_output) == int(item['label']):
            lr_status = "COR"
            overall_lr_correct += 1
        else:
            lr_status = "WRO"
    num_lr += 1
    if abs(lr_confidence) > lr_config.confidence_threshold:
        item_pred = lr_output
        by_lr += 1
        done_by = "LR"
    else:
        num_bert += 1
        if len(online_data_cache_for_bert['label']) == bert_config.online_batch_size:
            online_data = GenericDataset(online_data_cache_for_bert)
            online_dataloader = DataLoader(online_data, batch_size=bert_config.online_minibatch_size, shuffle=True)
            bert_model.train_online(online_dataloader)
            online_data_cache_for_bert["text"] = []
            online_data_cache_for_bert["label"] = []
            # print("BERT Updated")
            update_bert += 1
        bert_output, bert_confidence = bert_model.predict(item['text'])
        
        if int(bert_output) == int(item['label']):
            bert_status = "COR"
            overall_bert_correct += 1
        else:
            bert_status = "WRO"

        if max(bert_confidence) > bert_config.confidence_threshold:
            item_pred = bert_output
            by_bert += 1
            done_by = "BT"
        else:
            num_llama += 1
            done_by = "LM"
            llama_output, llama_confidence = llama_model.predict(PROMPT.format(item['text']))
            item_pred = postprocess(llama_output)
            if int(item_pred) == int(item['label']):
                llama_status = "COR"
                overall_llama_correct += 1
            else:
                llama_status = "WRO"

            online_data_cache_for_lr["text"].append(item['text'])
            online_data_cache_for_lr["label"].append(item_pred)
            online_data_cache_for_bert["text"].append(item['text'])
            online_data_cache_for_bert["label"].append(item_pred)
    
    if item_pred == int(item['label']):
        pipeline_correct += 1
        status = "COR"
    else:
        status = "WRO"
    
    msg = f"Idx {i+1} {status} by {done_by} | LR Pred: {llama_status}, Conf: {abs(lr_confidence):.2f}, Iter{update_lr} %: {by_lr / (i+1) :.2f}, Acc: {100 * overall_lr_correct / max(num_lr, 1) :.2f} | BT Pred: {bert_status}, Conf: {max(bert_confidence):.4f}, Iter{update_bert} %: {by_bert / (i+1) :.2f}, Acc: {100 * overall_bert_correct / max(num_bert, 1) :.2f} | LM Acc: {100 * overall_llama_correct / max(num_llama, 1) :.2f} | TT Acc: {100 * pipeline_correct / (i+1) :.2f}"
    f.write(msg + "\n")
    print(msg)
f.close()

