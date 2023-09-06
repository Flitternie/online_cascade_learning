import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
import datasets
from torch.utils.data import Dataset

from llama import inference
from utils import *

global DATASET
DATASET = 'hatespeech'

SytemPrompt = "You are given a post from an online forum and you need to check whether the post contains any hate speech. Return your answer in one word (yes or no) without any explanations. "
Prompt = '''{}'''
PROMPT = " ".join(["[INST]", B_SYS, SytemPrompt, E_SYS, Prompt, "[/INST]"])


class HateSpeechDataset(Dataset):
    def __init__(self, dataset):
        self.text = dataset['text']
        self.labels = dataset['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return PROMPT.format(self.text[i])


def main():
    data = datasets.load_from_disk('2000_sampled_hatespeech')
    dataset = HateSpeechDataset(data)
    print("Dataset length: ", len(dataset))

    # iterate through the agnews dataset 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    outputs = inference(dataloader)
    predictions = []
    for output in outputs:
        if "no" in output:
            predictions.append(0)
        elif "yes" in output:
            predictions.append(1)
        else:
            predictions.append(2)

    evaluate(dataset, outputs, predictions)
    print("Done!")

if __name__ == "__main__":
    main()