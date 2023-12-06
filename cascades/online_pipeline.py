import torch
import datasets
import numpy as np
import tqdm
import importlib

def pipeline(data_module, data, lr_wrapper, bert_wrapper, mu):
    lr_optimizer = torch.optim.Adam(lr_wrapper.parameters(), lr=lr_wrapper.learning_rate)
    bert_optimizer = torch.optim.Adam(bert_wrapper.parameters(), lr=bert_wrapper.learning_rate)
    mse_loss = torch.nn.MSELoss()
    cre_loss = torch.nn.CrossEntropyLoss()

    print("Data loaded, #labels: ", lr_wrapper.model.args.num_labels)
    f = open(f"./logs/online_cascade/{data_module.DATASET}/{mu:.4f}.log", "w+")

    lr_correct, bert_correct = 0, 0
    lr_score, bert_score = 0, 0
    lr_acted, bert_acted = 0, 0
    lr_update, bert_update = 0, 0
    llama_correct = 0
    llama_acted = 0
    overall_correct = 0

    lr_decision_correct, bert_decision_correct = 0, 0

    lr_wrapper.train()
    bert_wrapper.train()
    bar = tqdm.tqdm(range(len(data)))
    for i, item in enumerate(data):
        text = item['text']
        lr_pred, bert_pred, llama_pred = -1, -1, -1
        lr_decision, bert_decision = -1., -1.

        if len(lr_wrapper.model.online_cache['label']) == lr_wrapper.model.args.cache_size:
            lr_wrapper.model.train(lr_wrapper.model.online_cache)
            lr_wrapper.model.cache_clear()
            lr_update += 1
        if len(bert_wrapper.model.online_cache['label']) == bert_wrapper.model.args.cache_size:
            bert_wrapper.model.train_online(bert_wrapper.model.online_cache)
            bert_wrapper.model.cache_clear()
            bert_update += 1
        
        # warmup for the wrapper
        if lr_update < 10 or bert_update < 10:
            # llama_output, _ = llama_model.predict(PROMPT.format(item['text']))
            # llama_pred = postprocess(llama_output)
            llama_pred = int(item['llm_label'])
            if int(llama_pred) == item['label']:
                llama_correct += 1
                overall_correct += 1
            llama_acted += 1
            lr_wrapper.model.cache_add(text, llama_pred)
            bert_wrapper.model.cache_add(text, llama_pred)

            # warmup for the wrapper, skip the first updates
            if lr_update > 2 and bert_update > 2:
                lr_decision, lr_prob = lr_wrapper(text)
                lr_pred = torch.argmax(lr_prob, dim=-1).item()
                if int(lr_pred) == item['label']:
                    lr_correct += 1
                bert_decision, bert_prob = bert_wrapper(text)
                bert_pred = torch.argmax(bert_prob, dim=-1).item()
                if int(bert_pred) == item['label']:
                    bert_correct += 1

                lr_right_decision = torch.tensor([min(int(int(lr_pred) != llama_pred), 1.0)]).float().unsqueeze(0).to('cuda')
                lr_confidence_cost = mse_loss(lr_decision, lr_right_decision) 
                lr_confidence_cost += lr_wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in lr_wrapper.parameters()]))

                bert_right_decision = torch.tensor([min(int(int(bert_pred) != llama_pred), 1.0)]).float().unsqueeze(0).to('cuda')
                bert_confidence_cost = mse_loss(bert_decision, bert_right_decision)
                bert_confidence_cost += bert_wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in bert_wrapper.parameters()]))

                lr_cost = (1 - lr_decision.transpose(0,1)[-1]) * cre_loss(lr_prob, torch.tensor([llama_pred]).to('cuda')) + lr_decision.transpose(0,1)[-1] * lr_wrapper.model.args.cost * mu
                bert_cost = (1 - bert_decision.transpose(0,1)[-1]) * cre_loss(bert_prob, torch.tensor([llama_pred]).to('cuda')) + bert_decision.transpose(0,1)[-1] * bert_wrapper.model.args.cost * mu
                
                total_cost = lr_cost + bert_cost * lr_decision.transpose(0,1)[-1]

                lr_confidence_cost.backward(retain_graph=True)
                bert_confidence_cost.backward(retain_graph=True)
                total_cost.backward()
                lr_optimizer.step()
                bert_optimizer.step()
                lr_optimizer.zero_grad()
                bert_optimizer.zero_grad()
                lr_decision = float(lr_decision.squeeze(0)[-1].item())
                bert_decision = float(bert_decision.squeeze(0)[-1].item())
            
            f.write(f"{lr_pred},{bert_pred},{llama_pred},{item['label']},{lr_acted/(i+1):.2f},{bert_acted/(i+1):.2f},{lr_decision:.4f},{bert_decision:.4f},na,na,{lr_correct/(i+1):.2f},{bert_correct/(i+1):.2f},{llama_correct/max(1,llama_acted):.4f},{overall_correct/(i+1):.4f}\n")
            bar.update(1)
            continue

        # online learning process after warmup
        lr_decision, lr_prob = lr_wrapper(text)
        lr_pred = torch.argmax(lr_prob, dim=-1).item()
        if int(lr_pred) == item['label']:
            lr_correct += 1
        
        # decaying probability of proceeding to bert
        lr_decaying_prob = lr_wrapper.decaying_factor ** (lr_update)
        lr_action = (lr_decision.item() > 0.5)

        # for testing purpose! should be moved to line 155
        bert_decision, bert_prob = bert_wrapper(text)
        bert_pred = torch.argmax(bert_prob, dim=-1).item()
        if int(bert_pred) == item['label']:
            bert_correct += 1

        # logistic regression output
        if lr_action == 0 and np.random.choice([False,True], p=[lr_decaying_prob, 1-lr_decaying_prob]):
            if int(lr_pred) == item['label']:
                lr_score += 1
                overall_correct += 1
            lr_acted += 1
        # proceed to bert
        else:  
            # bert prediction

            # decaying probability of proceeding to llama
            bert_decaying_prob = bert_wrapper.decaying_factor ** (bert_update)
            bert_action = (bert_decision.item() > 0.5)

            # bert output
            if bert_action == 0 and np.random.choice([False,True], p=[bert_decaying_prob, 1-bert_decaying_prob]):
                if int(bert_pred) == item['label']:
                    bert_score += 1
                    overall_correct += 1
                bert_acted += 1
            # proceed to llama
            else:
                # llama_output, _ = llama_model.predict(PROMPT.format(item['text']))
                # llama_pred = postprocess(llama_output)
                llama_pred = int(item['llm_label'])
                if int(llama_pred) == item['label']:
                    llama_correct += 1
                    overall_correct += 1
                llama_acted += 1

                lr_wrapper.model.cache_add(text, llama_pred)
                bert_wrapper.model.cache_add(text, llama_pred)

                lr_decision_correct += int(int(lr_pred) == llama_pred) ^ int(lr_action != 0)
                bert_decision_correct += int(int(bert_pred) == llama_pred) ^ int(bert_action != 0)

                lr_right_decision = torch.tensor([min(int(int(lr_pred) != llama_pred) + lr_wrapper.calibration , 1.0)]).float().unsqueeze(0).to('cuda')
                lr_confidence_cost = mse_loss(lr_decision, lr_right_decision) 
                lr_confidence_cost += lr_wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in lr_wrapper.parameters()]))

                bert_right_decision = torch.tensor([min(int(int(bert_pred) != llama_pred) + bert_wrapper.calibration, 1.0)]).float().unsqueeze(0).to('cuda')
                bert_confidence_cost = mse_loss(bert_decision, bert_right_decision)
                bert_confidence_cost += bert_wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in bert_wrapper.parameters()]))

                lr_cost = (1 - lr_decision.transpose(0,1)[-1]) * cre_loss(lr_prob, torch.tensor([llama_pred]).to('cuda')) + lr_decision.transpose(0,1)[-1] * lr_wrapper.model.args.cost * mu
                bert_cost = (1 - bert_decision.transpose(0,1)[-1]) * cre_loss(bert_prob, torch.tensor([llama_pred]).to('cuda')) + bert_decision.transpose(0,1)[-1] * bert_wrapper.model.args.cost * mu
                
                total_cost = lr_cost + bert_cost * lr_decision.transpose(0,1)[-1]

                lr_confidence_cost.backward(retain_graph=True)
                bert_confidence_cost.backward(retain_graph=True)
                total_cost.backward()
                lr_optimizer.step()
                bert_optimizer.step()
                lr_optimizer.zero_grad()
                bert_optimizer.zero_grad()
                torch.cuda.empty_cache()

        bar.update(1)
        try:
            f.write(f"{lr_pred},{bert_pred},{llama_pred},{item['label']},{lr_acted/(i+1):.4f},{bert_acted/(i+1):.4f},{lr_decision:.4f},{bert_decision:.4f},{lr_score/max(1,lr_acted):.4f},{bert_score/max(1,bert_acted):.4f},{lr_correct/(i+1):.4f},{bert_correct/(i+1):.4f},{llama_correct/max(1,llama_acted):.4f},{overall_correct/(i+1):.4f}\n")
        except:
            try:
                lr_decision = float(lr_decision.squeeze(0)[-1].item())
                bert_decision = float(bert_decision.squeeze(0)[-1].item())
            except:
                pass
            f.write(f"{lr_pred},{bert_pred},{llama_pred},{item['label']},{lr_acted/(i+1):.4f},{bert_acted/(i+1):.4f},{lr_decision:.4f},{bert_decision:.4f},{lr_score/max(1,lr_acted):.4f},{bert_score/max(1,bert_acted):.4f},{lr_correct/(i+1):.4f},{bert_correct/(i+1):.4f},{llama_correct/max(1,llama_acted):.4f},{overall_correct/(i+1):.4f}\n")
        if i % 10 == 0:
            bar.set_description(f"{overall_correct/(i+1):.4f} | LR: {lr_update}:{lr_correct/(i+1):.4f} | {lr_score/max(1,lr_acted):.4f} , BERT: {bert_update}:{bert_correct/(i+1):.4f} | {bert_score/max(1,bert_acted):.4f} ")
    f.close()