import os
import torch
import datasets
import pickle
import numpy as np
import tqdm
import argparse

import lr
import bert

from inference_imdb import *
from utils import *
        
def main(mu):
    print("cost coefficient: ", mu)
    set_seed(42)
    data = datasets.load_from_disk("data/5000_sampled_imdb_test")
    data = data.shuffle()

    lr_config = ModelArguments()
    lr_config.num_labels = 2
    lr_config.cache_size = 8
    lr_config.cost = 1 #110M for bert-base
    lr_model = lr.LogisticRegressionModelSkLearn(lr_config, data=data['text'])
    lr_wrapper = ModelWrapper(lr_model, lr_model.args)
    lr_wrapper.to('cuda')
    
    bert_config = ModelArguments()
    bert_config.num_labels = 2
    bert_config.model = "bert-base-uncased"
    bert_config.cache_size = 16
    bert_config.batch_size = 8
    bert_config.num_epochs = 5
    bert_config.cost = 63 #7B for llama2-7b
    bert_model = bert.BertModel(bert_config)
    bert_wrapper = ModelWrapper(bert_model, bert_model.args)
    bert_wrapper.to('cuda')

    llama_config = ModelArguments()
    llama_config.model = "meta-llama/Llama-2-7b-chat-hf"
    llama_config.temperature = 0.3
    llama_config.max_new_tokens = 100
    llama_config.top_k = 5
    llama_model = llama.LlamaModel(llama_config)

    lr_optimizer = torch.optim.Adam(lr_wrapper.parameters(), lr=0.0007)
    bert_optimizer = torch.optim.Adam(bert_wrapper.parameters(), lr=0.0007)
    mse_loss = torch.nn.MSELoss()
    cre_loss = torch.nn.CrossEntropyLoss()

    print("Data loaded, #labels: ", lr_config.num_labels)
    f = open(f"logs/{DATASET}_online_2_{mu:.4f}.log", "w+")

    lr_correct, bert_correct = 0, 0
    lr_score, bert_score = 0, 0
    lr_acted, bert_acted = 0, 0
    lr_update, bert_update = 0, 0
    llama_correct = 0
    overall_correct = 0

    lr_decision_correct, bert_decision_correct = 0, 0

    lr_wrapper.train()
    bert_wrapper.train()
    bar = tqdm.tqdm(range(len(data)))
    for i, item in enumerate(data):
        text = item['text']

        llama_output, _ = llama_model.predict(PROMPT.format(item['text']))
        llama_pred = postprocess(llama_output)
        if int(llama_pred) == item['label']:
            llama_correct += 1

        lr_wrapper.model.cache_add(text, llama_pred)
        bert_wrapper.model.cache_add(text, llama_pred)

        if len(lr_wrapper.model.online_cache['label']) == lr_wrapper.model.args.cache_size:
            lr_wrapper.model.train(lr_wrapper.model.online_cache)
            lr_wrapper.model.cache_clear()
            lr_update += 1
        if len(bert_wrapper.model.online_cache['label']) == bert_wrapper.model.args.cache_size:
            bert_wrapper.model.train_online(bert_wrapper.model.online_cache)
            bert_wrapper.model.cache_clear()
            bert_update += 1

        if lr_update < 5:
            if int(llama_pred) == item['label']:
                overall_correct += 1
            f.write(f"na,na,{item['label']},{lr_acted/(i+1):.2f},{bert_acted/(i+1):.2f},na,na,{lr_correct/(i+1) :.4f},{bert_correct/(i+1) :.4f},{overall_correct/(i+1):.4f}\n")
            bar.update(1)
            continue

        lr_decision, lr_prob = lr_wrapper(text)
        lr_action = (lr_decision.item() > 0.5)
        lr_pred = torch.argmax(lr_prob, dim=-1).item()
        if int(lr_pred) == item['label']:
            lr_correct += 1
        
        bert_decision, bert_prob = bert_wrapper(text)
        bert_action = (bert_decision.item() > 0.5)
        bert_pred = torch.argmax(bert_prob, dim=-1).item()
        if int(bert_pred) == item['label']:
            bert_correct += 1

        if lr_action == 0:
            if int(lr_pred) == item['label']:
                lr_score += 1
                overall_correct += 1
            lr_acted += 1
        else:
            if bert_action == 0:
                if int(bert_pred) == item['label']:
                    bert_score += 1
                    overall_correct += 1
                bert_acted += 1
            else:
                if int(llama_pred) == item['label']:
                    overall_correct += 1
        

        lr_decision_correct += int(int(lr_pred) == item['label']) ^ int(lr_action != 0)
        bert_decision_correct += int(int(bert_pred) == item['label']) ^ int(bert_action != 0)

        lr_decay_factor = 0.5 ** (lr_update) + 0.4
        bert_decay_factor = 0.5 ** (bert_update) + 0.3

        lr_right_decision = torch.tensor([min(int(int(lr_pred) != item['label']) + lr_decay_factor, 1.0)]).float().unsqueeze(0).to('cuda')
        lr_confidence_cost = mse_loss(lr_decision, lr_right_decision) 
        lr_confidence_cost += 0.0001 * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in lr_wrapper.parameters()]))

        bert_right_decision = torch.tensor([min(int(int(bert_pred) != item['label']) + bert_decay_factor, 1.0)]).float().unsqueeze(0).to('cuda')
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
        # f.write(f"{lr_pred},{bert_pred},{item['label']},{lr_acted/(i+1):.2f},{bert_acted/(i+1):.2f},{lr_decision.squeeze(0)[-1].item():.4f},{bert_decision.squeeze(0)[-1].item():.4f},{lr_correct/(i+1) :.4f},{bert_correct/(i+1) :.4f},{overall_correct/(i+1):.4f}\n")
        f.write(f"{lr_pred},{bert_pred},{llama_pred},{item['label']},{lr_acted/(i+1):.2f},{bert_acted/(i+1):.2f},{lr_decision.squeeze(0)[-1].item():.4f},{bert_decision.squeeze(0)[-1].item():.4f},{lr_score/max(1,lr_acted):.4f},{bert_score/max(1,bert_acted):.4f},{lr_correct/(i+1):.4f},{bert_correct/(i+1):.4f},{llama_correct/(i+1):.4f},{overall_correct/(i+1):.4f}\n")
        
        if i % 50 == 0:
            bar.set_description(f"{overall_correct/(i+1):.4f} | LR: {lr_update}:{lr_correct/(i+1):.4f} | {lr_score/max(1,lr_acted):.4f} , BERT: {bert_update}:{bert_correct/(i+1):.4f} | {bert_score/max(1,bert_acted):.4f} ")
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.02)
    main(parser.parse_args().mu)