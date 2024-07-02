import os
import torch
import numpy as np
import tqdm
from typing import List, Dict, Any
from models import ModelWrapper

class OnlineCascade:
    """
    A class to manage and train a cascade of models online.

    Attributes:
        wrappers (List): List of model wrappers.
        mu (float): Regularization parameter.
        kwargs (Dict): Additional arguments for configuration.
        optimizers (List): List of optimizers for each model wrapper.
        mse_loss (torch.nn.Module): Mean Squared Error loss function.
        cre_loss (torch.nn.Module): Cross-Entropy loss function.
    """

    def __init__(self, wrappers: List[ModelWrapper], mu: float, **kwargs: Any) -> None:
        """
        Initialize the OnlineCascade class.

        Parameters:
            wrappers (List): List of model wrappers.
            mu (float): Regularization parameter.
            **kwargs (Dict): Additional arguments for configuration.
        """
        self.wrappers = wrappers
        self.mu = mu
        self.kwargs = kwargs

        self.optimizers = [torch.optim.Adam(wrapper.parameters(), lr=wrapper.learning_rate) for wrapper in wrappers]
        self.mse_loss = torch.nn.MSELoss()
        self.cre_loss = torch.nn.CrossEntropyLoss()

    def _setup_counter(self) -> None:
        """
        Setup counters for tracking model performance metrics.
        """
        self.model_correct = [0 for _ in range(len(self.wrappers))]
        self.model_score = [0 for _ in range(len(self.wrappers))]
        self.model_acted = [0 for _ in range(len(self.wrappers))]
        self.model_update = [0 for _ in range(len(self.wrappers))]
        self.model_decision_correct = [0 for _ in range(len(self.wrappers))]

        self.llm_correct = 0
        self.llm_acted = 0
        self.overall_correct = 0

    def _setup_logging(self) -> None:
        """
        Setup logging for model performance metrics.
        """
        wrapper_names = [wrapper.name for wrapper in self.wrappers]
        if 'log_dir' in self.kwargs:
            log_dir_path = os.path.join(self.kwargs['log_dir'], self.data_module.DATASET)
        elif 'config' in self.kwargs:
            log_dir_path = os.path.join(self.kwargs['config'].log, self.kwargs['config'].data.name)
        else:
            raise Exception("Log directory not declared")

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        log_file = os.path.join(log_dir_path, f"{'_'.join(wrapper_names)}_{self.kwargs['config'].llm.name}_{self.mu:.8f}.log")
        self.log_file = open(log_file, "w+", buffering=1)
        print(f"Writing to {log_file}")

    def _initialize_tracking_variables(self) -> None:
        """
        Initialize tracking variables for a new item.
        """
        self.llm_pred = -1
        self.model_preds = [-1 for _ in range(len(self.wrappers))]
        self.model_decisions = [-1.0 for _ in range(len(self.wrappers))]
        self.model_confidence_costs = [None for _ in range(len(self.wrappers))]
        self.model_costs = [None for _ in range(len(self.wrappers))]
        self.model_probs = [None for _ in range(len(self.wrappers))]
        self.model_decaying_probs = [0 for _ in range(len(self.wrappers))]
        self.model_actions = [1 for _ in range(len(self.wrappers))]

    def train_wrappers(self) -> None:
        """
        Set model wrappers to training mode.
        """
        for wrapper in self.wrappers:
            wrapper.train()

    def process_item(self, item: Dict[str, Any], idx: int) -> None:
        """
        Process a single data item.

        Parameters:
            item (Dict[str, Any]): The data item to process.
            idx (int): The index of the data item.
        """
        text = item['text']
        self._initialize_tracking_variables()  # Initialize tracking variables for the current item

        # Update model wrappers
        for j, wrapper in enumerate(self.wrappers):
            if len(wrapper.model.online_cache['llm_label']) == wrapper.model.args.cache_size:
                wrapper.model.train_online(wrapper.model.online_cache)
                self.model_update[j] += 1

        # Warm-up phase
        if min(self.model_update) < 10: # Hyperparameter: warm-up phase length
            self.llm_pred = int(item['llm_label'])  # LLM acts during warm-up
            acted_model_idx = len(self.wrappers)
            self.llm_acted += 1
            if int(self.llm_pred) == item['label']:
                self.llm_correct += 1
                self.overall_correct += 1

            if self.llm_pred != -1:
                # Add LLM annotation to model cache
                for wrapper in self.wrappers:
                    wrapper.model.cache_add(text, self.llm_pred)
                # Update wrappers if all models have been updated at least twice
                if min(self.model_update) > 2: # Hyperparameter: wrappers update before warm-up phase ends
                    self._warmup_wrappers(item)

            self._log_metrics(item, idx, acted_model_idx)
        else:
            acted_model_idx = self._online_learning(item)
            self._log_metrics(item, idx, acted_model_idx)

    def _warmup_wrappers(self, item: Dict[str, Any]) -> None:
        """
        Perform warm-up for model wrappers.

        Parameters:
            item (Dict[str, Any]): The data item being processed.
        """
        text = item['text']
        for j, wrapper in enumerate(self.wrappers):
            decision, prob = wrapper(text)
            pred = torch.argmax(prob, dim=-1).item()
            self.model_decisions[j] = decision
            self.model_preds[j] = pred
            if int(pred) == item['label']:
                self.model_correct[j] += 1
            # Set the decision label to 0 if the model prediction matches the LLM label and 1 otherwise
            model_right_decision = torch.tensor([min(int(int(pred) != self.llm_pred), 1.0)]).float().unsqueeze(0).to(wrapper.device)
            confidence_cost = self.mse_loss(decision, model_right_decision)
            confidence_cost += wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in wrapper.parameters()]))
            self.model_confidence_costs[j] = confidence_cost
            self.model_costs[j] = (1 - decision.transpose(0, 1)[-1]) * self.cre_loss(prob, torch.tensor([self.llm_pred]).to(wrapper.device)) + decision.transpose(0, 1)[-1] * wrapper.model.args.cost * self.mu

        total_cost = self.model_costs[0]
        for j in range(1, len(self.wrappers)):
            k = j - 1
            while k >= 0:
                self.model_costs[j] *= self.model_decisions[k].transpose(0, 1)[-1]
                k -= 1
            total_cost += self.model_costs[j]
        for j in range(len(self.wrappers)):
            self.model_confidence_costs[j].backward(retain_graph=True)
        total_cost.backward()
        for j in range(len(self.wrappers)):
            self.optimizers[j].step()
            self.optimizers[j].zero_grad()
            self.model_decisions[j] = float(self.model_decisions[j].squeeze(0)[-1].item())

    def _online_learning(self, item: Dict[str, Any]) -> int:
        """
        Perform online learning for model wrappers.

        Parameters:
            item (Dict[str, Any]): The data item being processed.

        Returns:
            int: Index of the model that acted.
        """
        text = item['text']
        for j, wrapper in enumerate(self.wrappers):
            decision, prob = wrapper(text)
            pred = torch.argmax(prob, dim=-1).item()
            self.model_decisions[j] = decision
            self.model_probs[j] = prob
            self.model_preds[j] = pred
            if int(pred) == item['label']:
                self.model_correct[j] += 1
            # Decay the probability of proceeding to the next level, set to higher values for less capable models, range [0,1)
            model_decayed_prob = wrapper.decaying_factor ** (self.model_update[j])
            self.model_decaying_probs[j] = model_decayed_prob
            model_action = (decision.item() > 0.5)
            self.model_actions[j] = model_action

        llm_acting = True
        acted_model_idx = -1
        for j, wrapper in enumerate(self.wrappers):
            # Stop proceeding to the next level if the model has already acted
            if self.model_actions[j] == 0 and np.random.choice([False, True], p=[self.model_decaying_probs[j], 1 - self.model_decaying_probs[j]]):
                if int(self.model_preds[j]) == item['label']:
                    self.model_score[j] += 1
                    self.overall_correct += 1
                self.model_acted[j] += 1
                acted_model_idx = j
                llm_acting = False
                break

        if llm_acting:
            acted_model_idx = self._handle_llm(item)

        return acted_model_idx

    def _handle_llm(self, item: Dict[str, Any]) -> int:
        """
        Handle LLM actions and update model caches.

        Parameters:
            item (Dict[str, Any]): The data item being processed.

        Returns:
            int: Index of the model that acted.
        """
        text = item['text']
        acted_model_idx = len(self.wrappers)
        self.llm_pred = int(item['llm_label'])
        if int(self.llm_pred) == item['label']:
            self.llm_correct += 1
            self.overall_correct += 1
        self.llm_acted += 1

        if self.llm_pred != -1:
            for j, wrapper in enumerate(self.wrappers):
                wrapper.model.cache_add(text, self.llm_pred)
                # Increment decision_correct if the model's prediction is correct and it took an action
                self.model_decision_correct[j] += int(int(self.model_preds[j]) == self.llm_pred) ^ int(self.model_actions[j] != 0)
                # wrapper.calibration is to adjust the model prediction's confidence, set to higher values for less capable models, range [0,0.5)
                model_right_decision = torch.tensor([min(int(int(self.model_preds[j]) != self.llm_pred) + wrapper.calibration, 1.0)]).float().unsqueeze(0).to(wrapper.device)
                confidence_cost = self.mse_loss(self.model_decisions[j], model_right_decision)
                confidence_cost += wrapper.regularization * torch.sum(torch.stack([torch.sum(torch.square(param)) for param in wrapper.parameters()]))
                self.model_confidence_costs[j] = confidence_cost
                self.model_costs[j] = (1 - self.model_decisions[j].transpose(0, 1)[-1]) * self.cre_loss(self.model_probs[j], torch.tensor([self.llm_pred]).to(wrapper.device)) + self.model_decisions[j].transpose(0, 1)[-1] * wrapper.model.args.cost * self.mu

            total_cost = self.model_costs[0]
            for j in range(1, len(self.wrappers)):
                k = j - 1
                while k >= 0:
                    self.model_costs[j] *= self.model_decisions[k].transpose(0, 1)[-1]
                    k -= 1
                total_cost += self.model_costs[j]
            for j in range(len(self.wrappers)):
                self.model_confidence_costs[j].backward(retain_graph=True)
            total_cost.backward()
            for j in range(len(self.wrappers)):
                self.optimizers[j].step()
                self.optimizers[j].zero_grad()
                self.model_decisions[j] = float(self.model_decisions[j].squeeze(0)[-1].item())

        torch.cuda.empty_cache()
        return acted_model_idx

    def _log_metrics(self, item: Dict[str, Any], idx: int, acted_model_idx: int) -> None:
        """
        Log performance metrics to file.

        Parameters:
            item (Dict[str, Any]): The data item being processed.
            idx (int): The index of the data item.
            acted_model_idx (int): Index of the model that acted.
        """
        output = ""
        for j in range(len(self.wrappers)):
            output += f"{self.model_preds[j]},"
        output += f"{self.llm_pred},{item['label']},"
        for j in range(len(self.wrappers)):
            output += f"{self.model_acted[j]/(idx+1):.4f},"
        for j in range(len(self.wrappers)):
            try:
                output += f"{self.model_decisions[j]:.4f},"
            except:
                # In case model_decisions is a tensor
                self.model_decisions[j] = float(self.model_decisions[j].squeeze(0)[-1].item())
                output += f"{self.model_decisions[j]:.4f},"
        for j in range(len(self.wrappers)):
            output += f"{self.model_score[j]/max(1, self.model_acted[j]):.4f},"
        for j in range(len(self.wrappers)):
            output += f"{self.model_correct[j]/(idx+1):.4f},"
        output += f"{self.llm_correct/max(1, self.llm_acted):.4f},{self.overall_correct/(idx+1):.4f},"
        assert acted_model_idx >= 0 and acted_model_idx <= len(self.wrappers)
        output += f"{acted_model_idx}"

        if hasattr(self.data_module, 'CustomMetrics'):
            self.data_custom_metrics.update(self.model_acted, self.llm_acted, self.model_preds, self.llm_pred, item['label'])
            custom_metrics = self.data_custom_metrics.get_metrics()
            for idx, results in enumerate(custom_metrics[:len(self.wrappers)]):
                output += f", {self.wrappers[idx].name}: " + ";".join([f"{r:.4f}" for r in results])
            output += f", LLM: " + ";".join([f"{r:.4f}" for r in custom_metrics[-2]])
            output += f", Overall: " + ";".join([f"{r:.4f}" for r in custom_metrics[-1]])

        output += "\n"
        self.log_file.write(output)

    def run(self, data: List[Dict[str, Any]], data_module: Any) -> None:
        """
        Run the online cascade process.

        Parameters:
            data (List[Dict[str, Any]]): The dataset to process.
            data_module (Any): The data module containing custom metrics and configurations.
        """
        self.data = data
        self.data_module = data_module
        if hasattr(data_module, 'CustomMetrics'):
            self.data_custom_metrics = data_module.CustomMetrics(len(self.wrappers))

        self._setup_logging()
        self._setup_counter()

        self.train_wrappers()
        bar = tqdm.tqdm(range(len(self.data)))
        for idx, item in enumerate(self.data):
            self.process_item(item, idx)
            bar.update(1)
            # Set description for progress bar
            if idx % 10 == 0:
                description = ""
                if hasattr(self.data_module, 'CustomMetrics'):
                    custom_metrics = self.data_custom_metrics.get_short_metrics()
                else:
                    custom_metrics = None
                description += f"{self.overall_correct/(idx+1):.4f} |"
                for j, wrapper in enumerate(self.wrappers):
                    description += f" {wrapper.name}: {self.model_update[j]}:{self.model_correct[j]/(idx+1):.4f},"
                    description += f"{self.model_score[j]/max(1, self.model_acted[j]):.4f}"
                    if custom_metrics:
                        description += f",{custom_metrics[j][0]:.4f}"
                    description += " |"
                bar.set_description(description)
        # Close log file
        self.log_file.close()


def pipeline(data_module, data, wrappers, mu, **kwargs):
    optimizers = [torch.optim.Adam(wrapper.parameters(), lr=wrapper.learning_rate) for wrapper in wrappers]
    mse_loss = torch.nn.MSELoss()
    cre_loss = torch.nn.CrossEntropyLoss()

    # check if args.num_labels is the same for all wrappers
    assert len(set([wrapper.model.args.num_labels for wrapper in wrappers])) == 1
    print("Data loaded, #labels: ", wrappers[0].model.args.num_labels)
    wrapper_names = [wrapper.name for wrapper in wrappers]
    
    if 'log_dir' in kwargs.keys():
        log_dir_path = os.path.join(kwargs['log_dir'], data_module.DATASET)
        # check if directory exists, create if not
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        log_file = os.path.join(log_dir_path, f"{'_'.join(wrapper_names)}_{mu:.8f}.log")
    elif 'config' in kwargs.keys():
        log_dir_path = os.path.join(kwargs['config'].log, kwargs['config'].data.name)
        # check if directory exists, create if not
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        log_file = os.path.join(log_dir_path, f"{'_'.join(wrapper_names)}_{kwargs['config'].llm.name}_{mu:.8f}.log")
    else:
        raise Exception("log directory not declared")
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