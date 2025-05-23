from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from datasets import Dataset

DATASET = 'imdb'

SystemPrompt = "You are a helpful, respectful and honest assistant. The user has given you a movie review to help them make their decision. You should read the review and tell the user whether the review overall shows a positive or negative sentiment towards the movie. Return the answer in one word. "
UserPrompt = '''Here is the movie review: {} \n Tell me whether the above review overall shows a positive or negative sentiment towards the movie. Return the answer in one word.'''
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, UserPrompt, "[/INST]"])

LabelList = ['positive', 'negative']

def preprocess(data: Dataset) -> Dataset:
    return data

def postprocess(output: str) -> int:
    output = output.lower().strip()
    try:
        return int(output)
    except ValueError:
        if "positive" in output:
            return 1
        elif "negative" in output:
            return 0
        else:
            return -1