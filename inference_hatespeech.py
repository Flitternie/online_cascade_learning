import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
import datasets
from torch.utils.data import Dataset

import argparse
import llama
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

def postprocess(output):
    if "no" in output:
        return 0
    elif "yes" in output:
        return 1
    else:
        return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/2000_sampled_hatespeech")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = datasets.load_from_disk(args.data_dir)
    dataset = HateSpeechDataset(data)
    print("Dataset length: ", len(dataset))

    # iterate through the agnews dataset 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    llama_model = llama.LlamaModel(args)
    outputs = llama_model.inference(dataloader)
    predictions = []
    for output in outputs:
        predictions.append(postprocess(output))

    evaluate(dataset, outputs, predictions)
    print("Done!")

if __name__ == "__main__":
    main()