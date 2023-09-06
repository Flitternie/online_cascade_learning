import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
import datasets
from torch.utils.data import Dataset

from llama import inference
from utils import *

global DATASET
DATASET = 'imdb'

SytemPrompt = "You are a helpful, respectful and honest assistant. The user has given you a movie review to help them make their decision. You should read the review and tell the user whether the review overall shows a positive or negative sentiment towards the movie. Return the answer in one word. "
Prompt = '''Here is the movie review: {} \n Tell me whether the above review overall shows a positive or negative sentiment towards the movie. Return the answer in one word.'''
PROMPT = " ".join(["[INST]", B_SYS, SytemPrompt, E_SYS, Prompt, "[/INST]"])


class ImdbDataset(Dataset):
    def __init__(self, dataset):
        self.text = dataset['text']
        self.labels = dataset['label']

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return PROMPT.format(self.text[i])


def main():
    data = datasets.load_from_disk('1000_sampled_imdb')
    dataset = ImdbDataset(data)
    print("Dataset length: ", len(dataset))

    # iterate through the agnews dataset 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    outputs = inference(dataloader)
    predictions = []
    for output in outputs:
        if "positive" in output:
            predictions.append(1)
        elif "negative" in output:
            predictions.append(0)
        else:
            predictions.append(2)

    evaluate(dataset, outputs, predictions)
    print("Done!")

if __name__ == "__main__":
    main()