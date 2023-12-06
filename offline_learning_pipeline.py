import os
import torch
import datasets
import pickle
import numpy as np
import tqdm

import bert

from inference_imdb import *
from utils import *
        
def main(mu):
    print("cost coefficient: ", mu)
    lr_config = ModelArguments()
    lr_config.num_labels = 2
    lr_config.cache_size = 8
    lr_config.cost = 1 #110M for bert-base
    lr_model = pickle.load(open("models/lr_imdb_5000_torch.pkl", "rb"))
    lr_model.args = update_config(lr_model.args, lr_config)
    lr_wrapper = ModelWrapper(lr_model, lr_model.args)
    lr_wrapper.to('cuda')
    
    bert_config = ModelArguments()
    bert_config.num_labels = 2
    bert_config.model = "./models/bert_imdb_5000"
    bert_config.cache_size = 16
    bert_config.cost = 63 #7B for llama2-7b
    bert_model = bert.BertModel(bert_config)
    bert_wrapper = ModelWrapper(bert_model, bert_model.args)
    bert_wrapper.to('cuda')

    llama_config = ModelArguments()
    llama_config.model = "meta-llama/Llama-2-7b-chat-hf"
    llama_config.temperature = 0.3
    llama_config.max_new_tokens = 100
    llama_config.top_k = 5
    # llama_model = llama.LlamaModel(llama_config)

    lr_optimizer = torch.optim.Adam(lr_wrapper.parameters(), lr=0.0007)
    bert_optimizer = torch.optim.Adam(bert_wrapper.parameters(), lr=0.0007)
    mse_loss = torch.nn.MSELoss()
    cre_loss = torch.nn.CrossEntropyLoss()
    set_seed(42)
    data = datasets.load_from_disk("data/5000_sampled_imdb_test")
    data = data.shuffle()

    print("Data loaded, #labels: ", lr_config.num_labels)
    f = open(f"logs/{DATASET}_post_3_{mu:.4f}_offline.log", "w+")

    lr_correct, bert_correct = 0, 0
    lr_score, bert_score = 0, 0
    lr_acted, bert_acted = 0, 0
    overall_correct = 0

    lr_decision_correct, bert_decision_correct = 0, 0

    lr_wrapper.train()
    bert_wrapper.train()
    bar = tqdm.tqdm(range(len(data)))
    for i, item in enumerate(data):
        text = item['text']
        lr_decision, lr_prob = lr_wrapper(text)
        # lr_action = torch.argmax(lr_decision, dim=-1).item()
        lr_action = (lr_decision.item() > 0.5)
        lr_pred = torch.argmax(lr_prob, dim=-1).item()
        
        bert_decision, bert_prob = bert_wrapper(text)
        # bert_action = torch.argmax(bert_decision, dim=-1).item()
        bert_action = (bert_decision.item() > 0.5)
        bert_pred = torch.argmax(bert_prob, dim=-1).item()

        if int(lr_pred) == item['label']:
            lr_correct += 1
        if lr_action == 0:
            if int(lr_pred) == item['label']:
                lr_score += 1
                overall_correct += 1
            lr_acted += 1
            
        if int(bert_pred) == item['label']:
            bert_correct += 1
        if lr_action != 0 and bert_action == 0:
            if int(bert_pred) == item['label']:
                bert_score += 1
                overall_correct += 1
            bert_acted += 1         
        elif lr_action != 0 and bert_action != 0:
            overall_correct += 1

        lr_decision_correct += int(int(lr_pred) == item['label']) ^ int(lr_action != 0)
        bert_decision_correct += int(int(bert_pred) == item['label']) ^ int(bert_action != 0)

        lr_right_decision = torch.tensor([min(int(int(lr_pred) != item['label']) + 0.3, 1.0)]).float().unsqueeze(0).to('cuda')
        lr_confidence_cost = mse_loss(lr_decision, lr_right_decision) 
        lr_confidence_cost += 0.0001 * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in lr_wrapper.parameters()]))

        bert_right_decision = torch.tensor([int(int(bert_pred) != item['label'])]).float().unsqueeze(0).to('cuda')
        bert_confidence_cost = mse_loss(bert_decision, bert_right_decision)
        bert_confidence_cost += 0.0001 * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in bert_wrapper.parameters()]))

        lr_cost = (1 - lr_decision.transpose(0,1)[-1]) * cre_loss(lr_prob, torch.tensor([item['label']]).to('cuda')) + lr_decision.transpose(0,1)[-1] * lr_wrapper.model.args.cost * mu
        bert_cost = (1 - bert_decision.transpose(0,1)[-1]) * cre_loss(bert_prob, torch.tensor([item['label']]).to('cuda')) + bert_decision.transpose(0,1)[-1] * bert_wrapper.model.args.cost * mu
        
        total_cost = lr_cost + bert_cost * lr_decision.transpose(0,1)[-1]

        lr_confidence_cost.backward(retain_graph=True)
        bert_confidence_cost.backward(retain_graph=True)
        total_cost.backward()
        lr_optimizer.step()
        bert_optimizer.step()
        lr_optimizer.zero_grad()
        bert_optimizer.zero_grad()
        bar.update(1)
        # f.write(f"{int(lr_pred) == item['label']}, {lr_prob.tolist()}, {lr_decision.transpose(0,1)[-1].item():.4f} | {int(bert_pred) == item['label']}, {bert_prob.tolist()}, {bert_decision.transpose(0,1)[-1].item():.4f}\n")
        f.write(f"{lr_pred},{bert_pred},{item['label']},{lr_acted/(i+1):.2f},{bert_acted/(i+1):.2f},{lr_decision.squeeze(0)[-1].item():.4f},{bert_decision.squeeze(0)[-1].item():.4f},{lr_correct/(i+1) :.4f},{bert_correct/(i+1) :.4f},{overall_correct/(i+1):.4f}\n")
        if i % 100 == 0:
            bar.set_description(f"{overall_correct/(i+1) :.4f} | LR: {lr_correct/(i+1) :.4f} | {lr_decision_correct/(i+1) : .4f} , BERT: {bert_correct/(i+1) :.4f} | {bert_decision_correct/(i+1) : .4f} ")
    f.close()

if __name__ == "__main__":
    for mu in np.arange(0.01, 0.06, 0.01):
        main(mu)