from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from datasets import Dataset

DATASET = 'hatespeech'

SystemPrompt = "You are given a post from an online forum and you need to check whether the post contains any hate speech. Return your answer in one word (yes or no) without any explanations. "
UserPrompt = '''Post: {}'''
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, UserPrompt, "[/INST]"])

LabelList = ['yes', 'no']

def preprocess(data: Dataset) -> Dataset:
    return data

def postprocess(output: str) -> int:
    output = output.lower().strip()
    try:
        return int(output)
    except ValueError:
        if "no" in output:
            return 0
        elif "yes" in output:
            return 1
        else:
            return -1

class CustomMetrics():
    # return recall, precision, f1 for each of the models
    def __init__(self, num_models: int) -> None:
        self.num_models = num_models + 1 # +1 for llm
        self.true_pos = [0] * self.num_models
        self.false_pos = [0] * self.num_models
        self.false_neg = [0] * self.num_models
        self.model_acted_counter = [0] * (self.num_models - 1)
        self.llm_acted_counter = 0

        self.overall_true_pos = 0
        self.overall_false_pos = 0
        self.overall_false_neg = 0
        
    def update(self, model_acted: list, llm_acted: int, model_preds: list, llm_label: int, label: int) -> None:
        for i, m in enumerate(model_preds):
            if m == 1 and label == 1:
                self.true_pos[i] += 1
            elif m == 1 and label == 0:
                self.false_pos[i] += 1
            elif m == 0 and label == 1:
                self.false_neg[i] += 1
        if llm_label == 1 and label == 1:
            self.true_pos[-1] += 1
        elif llm_label == 1 and label == 0:
            self.false_pos[-1] += 1
        elif llm_label == 0 and label == 1:
            self.false_neg[-1] += 1    
        # determine which model acted based on the increment in llm_acted & model_acted
        if llm_acted != self.llm_acted_counter:
            # llm acted
            if llm_label == 1 and label == 1:
                self.overall_true_pos += 1
            elif llm_label == 1 and label == 0:
                self.overall_false_pos += 1
            elif llm_label == 0 and label == 1:
                self.overall_false_neg += 1
        elif model_acted != self.model_acted_counter:
            # check which model acted by comparing model_acted_counter and model_acted
            for i in range(len(model_acted)):
                if model_acted[i] != self.model_acted_counter[i]:
                    # model acted
                    if model_preds[i] == 1 and label == 1:
                        self.overall_true_pos += 1
                    elif model_preds[i] == 1 and label == 0:
                        self.overall_false_pos += 1
                    elif model_preds[i] == 0 and label == 1:
                        self.overall_false_neg += 1
                    break
        else:
            raise Exception("No model acted")
        # set to same value but not pointer
        self.model_acted_counter = model_acted.copy()
        self.llm_acted_counter = llm_acted

    def get_metrics(self) -> list:
        metrics = []
        for i in range(self.num_models):
            if self.true_pos[i] + self.false_pos[i] == 0:
                precision = 0
            else:
                precision = self.true_pos[i] / (self.true_pos[i] + self.false_pos[i])
            if self.true_pos[i] + self.false_neg[i] == 0:
                recall = 0
            else:
                recall = self.true_pos[i] / (self.true_pos[i] + self.false_neg[i])
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            metrics.append((precision, recall, f1))
        # overall metrics
        if self.overall_true_pos + self.overall_false_pos == 0:
            overall_precision = 0
        else:
            overall_precision = self.overall_true_pos / (self.overall_true_pos + self.overall_false_pos)
        if self.overall_true_pos + self.overall_false_neg == 0:
            overall_recall = 0
        else:
            overall_recall = self.overall_true_pos / (self.overall_true_pos + self.overall_false_neg)
        if overall_precision + overall_recall == 0:
            overall_f1 = 0
        else:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)

        return metrics + [(overall_precision, overall_recall, overall_f1)]
    
    def get_short_metrics(self) -> list:
        metrics = self.get_metrics()
        short_metrics = []
        # get only recall
        for m in metrics:
            short_metrics.append([m[1]])
        return short_metrics