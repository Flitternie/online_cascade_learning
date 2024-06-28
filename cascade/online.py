import os
import torch
import numpy as np
import tqdm

def pipeline(data_module, data, wrappers, mu):
    optimizers = [torch.optim.Adam(wrapper.parameters(), lr=wrapper.learning_rate) for wrapper in wrappers]
    mse_loss = torch.nn.MSELoss()
    cre_loss = torch.nn.CrossEntropyLoss()

    # check if args.num_labels is the same for all wrappers
    assert len(set([wrapper.model.args.num_labels for wrapper in wrappers])) == 1
    print("Data loaded, #labels: ", wrappers[0].model.args.num_labels)
    wrapper_names = [wrapper.name for wrapper in wrappers]
    # check if directory exists, create if not
    if not os.path.exists(f"./logs/online_cascade_general/{data_module.DATASET}"):
        os.makedirs(f"./logs/online_cascade_general/{data_module.DATASET}")
    log_file = f"./logs/online_cascade_general/{data_module.DATASET}/{'_'.join(wrapper_names)}_{mu:.8f}.log"
    f = open(log_file, "w+")
    print(f"Writing to {log_file}")

    model_correct = [0 for _ in range(len(wrappers))]
    model_score = [0 for _ in range(len(wrappers))]
    model_acted = [0 for _ in range(len(wrappers))]
    model_update = [0 for _ in range(len(wrappers))]
    model_decision_correct = [0 for _ in range(len(wrappers))]
    
    llm_correct = 0
    llm_acted = 0
    overall_correct = 0

    if hasattr(data_module, 'CustomMetrics'):
        data_custom_metrics = data_module.CustomMetrics(len(wrappers))

    for wrapper in wrappers:
        wrapper.train()

    bar = tqdm.tqdm(range(len(data)))
    for i, item in enumerate(data):
        text = item['text']
        model_preds = [-1 for _ in range(len(wrappers))]
        llm_pred = -1
        model_decisions = [-1. for _ in range(len(wrappers))]

        for j, wrapper in enumerate(wrappers):
            if len(wrapper.model.online_cache['llm_label']) == wrapper.model.args.cache_size:
                wrapper.model.train_online(wrapper.model.online_cache)
                wrapper.model.cache_clear()
                model_update[j] += 1
        
        model_confidence_costs = [None for _ in range(len(wrappers))]
        model_costs = [None for _ in range(len(wrappers))]
        model_probs = [None for _ in range(len(wrappers))]
        model_decaying_probs = [0 for _ in range(len(wrappers))]
        model_actions = [1 for _ in range(len(wrappers))]

        # warm-up wrappers
        if min(model_update) < 10:
            llm_pred = int(item['llm_label'])
            if int(llm_pred) == item['label']:
                llm_correct += 1
                overall_correct += 1
            llm_acted += 1
            if llm_pred != -1:
                for wrapper in wrappers:            
                    wrapper.model.cache_add(text, llm_pred)

                # warm-up wrappers, skip the first updates
                if min(model_update) > 2:
                    for j, wrapper in enumerate(wrappers):
                        decision, prob = wrapper(text)
                        model_decisions[j] = decision
                        pred = torch.argmax(prob, dim=-1).item()
                        model_preds[j] = pred
                        if int(pred) == item['label']:
                            model_correct[j] += 1
                        # set the decision to 0 if the model prediction matches the LLM label and 1 otherwise
                        model_right_decision = torch.tensor([min(int(int(pred) != llm_pred), 1.0)]).float().unsqueeze(0).to(wrapper.device)
                        confidence_cost = mse_loss(decision, model_right_decision) 
                        confidence_cost += wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in wrapper.parameters()]))
                        model_confidence_costs[j] = confidence_cost
                        model_costs[j] = (1 - decision.transpose(0,1)[-1]) * cre_loss(prob, torch.tensor([llm_pred]).to(wrapper.device)) + decision.transpose(0,1)[-1] * wrapper.model.args.cost * mu
                    
                    total_cost = model_costs[0]
                    for j in range(1, len(wrappers)):
                        k = j - 1
                        while k >= 0: 
                            model_costs[j] *= model_decisions[k].transpose(0,1)[-1]
                            k -= 1
                        total_cost += model_costs[j]
                    for j in range(len(wrappers)):
                        model_confidence_costs[j].backward(retain_graph=True)
                    total_cost.backward()
                    for j in range(len(wrappers)):
                        optimizers[j].step()
                        optimizers[j].zero_grad()
                        model_decisions[j] = float(model_decisions[j].squeeze(0)[-1].item())
            
            output = ''''''
            for j, wrapper in enumerate(wrappers):
                output += f"{model_preds[j]},"
            output += f"{llm_pred},{item['label']},"
            for j, wrapper in enumerate(wrappers):
                output += f"{model_acted[j]/(i+1):.4f},"
            for j, wrapper in enumerate(wrappers):
                output += f"{model_decisions[j]:.4f},"
            for j, wrapper in enumerate(wrappers):
                output += f"{model_score[j]/max(1,model_acted[j]):.4f},"
            for j, wrapper in enumerate(wrappers):
                output += f"{model_correct[j]/(i+1):.4f},"
            output += f"{llm_correct/max(1,llm_acted):.4f},{overall_correct/(i+1):.4f}"

            if hasattr(data_module, 'CustomMetrics'):
                data_custom_metrics.update(model_acted, llm_acted, model_preds, llm_pred, item['label'])
                custom_metrics = data_custom_metrics.get_metrics()
                for idx, results in enumerate(custom_metrics[:len(wrappers)]):
                    output += f", {wrappers[idx].name}: " + ";".join([f"{r:.4f}" for r in results])
                output += f", LLM: " + ";".join([f"{r:.4f}" for r in custom_metrics[-2]])
                output += f", Overall: " + ";".join([f"{r:.4f}" for r in custom_metrics[-1]])
            
            output += "\n"
            f.write(output)
            bar.update(1)
            continue

        # online learning process after warmup
        for j, wrapper in enumerate(wrappers):
            decision, prob = wrapper(text)
            pred = torch.argmax(prob, dim=-1).item()
            model_decisions[j] = decision
            model_probs[j] = prob
            model_preds[j] = pred
            if int(pred) == item['label']:
                model_correct[j] += 1
            # decay the probability of proceeding to the next level, set to higher values for less capable models, range [0,1)
            model_decayed_prob = wrapper.decaying_factor ** (model_update[j])
            model_decaying_probs[j] = model_decayed_prob
            model_action = (decision.item() > 0.5)
            model_actions[j] = model_action
        
        llm_flag = True
        acted_model_idx = -1
        for j, wrapper in enumerate(wrappers):
            # stop proceeding to the next level if the model has already acted
            if model_actions[j] == 0 and np.random.choice([False,True], p=[model_decaying_probs[j], 1-model_decaying_probs[j]]):
                if int(model_preds[j]) == item['label']:
                    model_score[j] += 1
                    overall_correct += 1
                model_acted[j] += 1
                acted_model_idx = j
                llm_flag = False
                break
        
        # proceed to llm
        if llm_flag:
            acted_model_idx = len(wrappers)
            llm_pred = int(item['llm_label'])
            if int(llm_pred) == item['label']:
                llm_correct += 1
                overall_correct += 1
            llm_acted += 1

            if llm_pred != -1:            
                for j, wrapper in enumerate(wrappers):
                    wrapper.model.cache_add(text, llm_pred)
                    # increment decision_correct if the model's prediction is correct and it took an action
                    model_decision_correct[j] += int(int(model_preds[j]) == llm_pred) ^ int(model_actions[j] != 0)
                    # wrapper.calibration is to adjust the model prediction's confidence, set to higher values for less capable models, range [0,0.5)
                    model_right_decision = torch.tensor([min(int(int(model_preds[j]) != llm_pred) + wrapper.calibration, 1.0)]).float().unsqueeze(0).to(wrapper.device)
                    model_confidence_cost = mse_loss(model_decisions[j], model_right_decision)
                    model_confidence_cost += wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in wrapper.parameters()]))
                    model_confidence_costs[j] = model_confidence_cost

                    model_cost = (1 - model_decisions[j].transpose(0,1)[-1]) * cre_loss(model_probs[j], torch.tensor([llm_pred]).to(wrapper.device)) + model_decisions[j].transpose(0,1)[-1] * wrapper.model.args.cost * mu
                    model_costs[j] = model_cost

                total_cost = model_costs[0]
                for j in range(1, len(wrappers)):
                    k = j - 1
                    while k >= 0: 
                        model_costs[j] *= model_decisions[k].transpose(0,1)[-1]
                        k -= 1
                    total_cost += model_costs[j]
                for j in range(len(wrappers)):
                    model_confidence_costs[j].backward(retain_graph=True)
                total_cost.backward()
                for j in range(len(wrappers)):
                    optimizers[j].step()
                    optimizers[j].zero_grad()
                    model_decisions[j] = float(model_decisions[j].squeeze(0)[-1].item())
                
            torch.cuda.empty_cache()

        output = ''''''
        for j, wrapper in enumerate(wrappers):
            output += f"{model_preds[j]},"
        output += f"{llm_pred},{item['label']},"
        for j, wrapper in enumerate(wrappers):
            output += f"{model_acted[j]/(i+1):.4f},"
        for j, wrapper in enumerate(wrappers):
            try:
                output += f"{model_decisions[j]:.4f},"
            except:
                model_decisions[j] = float(model_decisions[j].squeeze(0)[-1].item())
                output += f"{model_decisions[j]:.4f},"
        for j, wrapper in enumerate(wrappers):
            output += f"{model_score[j]/max(1,model_acted[j]):.4f},"
        for j, wrapper in enumerate(wrappers):
            output += f"{model_correct[j]/(i+1):.4f},"
        output += f"{llm_correct/max(1,llm_acted):.4f},{overall_correct/(i+1):.4f},"
        assert acted_model_idx >= 0 and acted_model_idx <= len(wrappers)
        output += f"{acted_model_idx}"

        if hasattr(data_module, 'CustomMetrics'):
            data_custom_metrics.update(model_acted, llm_acted, model_preds, llm_pred, item['label'])
            custom_metrics = data_custom_metrics.get_metrics()
            for idx, results in enumerate(custom_metrics[:len(wrappers)]):
                output += f", {wrappers[idx].name}: " + ";".join([f"{r:.4f}" for r in results])
            output += f", LLM: " + ";".join([f"{r:.4f}" for r in custom_metrics[-2]])
            output += f", Overall: " + ";".join([f"{r:.4f}" for r in custom_metrics[-1]])
        
        output += "\n"
        f.write(output)
        bar.update(1)
        
        if i % 10 == 0:
            description = ''''''
            if hasattr(data_module, 'CustomMetrics'):
                custom_metrics = data_custom_metrics.get_short_metrics()
            else:
                custom_metrics = None
            description += f"{overall_correct/(i+1):.4f} |"
            for j, wrapper in enumerate(wrappers):
                description += f" {wrapper.name}: {model_update[j]}:{model_correct[j]/(i+1):.4f},"
                description += f"{model_score[j]/max(1,model_acted[j]):.4f}"
                if custom_metrics:
                    description += f",{custom_metrics[j][0]:.4f}"
                description += " |"
            
            bar.set_description(description)
    f.close()