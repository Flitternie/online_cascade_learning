import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
import datasets
from torch.utils.data import Dataset

from llama import inference
from utils import *

global DATASET
DATASET = 'agnews'

SytemPrompt = "You are given a sentence that belongs to one of the 4 categories including (1) World, (2) Sports, (3) Business, (4) Sci/Tech. You should read the sentence and tell the user which category it belongs to. Return your answer in one word. "
Prompt = '''{}'''
PROMPT = " ".join(["[INST]", B_SYS, SytemPrompt, E_SYS, Prompt, "[/INST]"])


class AGNewsDataset(Dataset):
    def __init__(self, dataset):
        self.text = dataset['text']
        self.labels = dataset['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return PROMPT.format(self.text[i])

def main():
    data = datasets.load_from_disk('2000_sampled_agnews')
    dataset = AGNewsDataset(data)
    print("Dataset length: ", len(dataset))

    # iterate through the agnews dataset 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    outputs = inference(dataloader)
    predictions = []
    for output in outputs:
        if "world" in output:
            predictions.append(0)
        elif "sports" in output:
            predictions.append(1)
        elif "business" in output:
            predictions.append(2)
        elif "sci" in output or "tech" in output:
            predictions.append(3)
        else:
            predictions.append(4)

    evaluate(dataset, outputs, predictions)
    print("Done!")

if __name__ == "__main__":
    main()