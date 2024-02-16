from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from utils import *

DATASET = 'imdb'

SystemPrompt = "You are a helpful, respectful and honest assistant. The user has given you a movie review to help them make their decision. You should read the review and tell the user whether the review overall shows a positive or negative sentiment towards the movie. Return the answer in one word. "
Prompt = '''Here is the movie review: {} \n Tell me whether the above review overall shows a positive or negative sentiment towards the movie. Return the answer in one word.'''
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, Prompt, "[/INST]"])

def postprocess(output: str) -> int:
    output = output.lower().strip()
    if "positive" in output:
        return 1
    elif "negative" in output:
        return 0
    else:
        return -1