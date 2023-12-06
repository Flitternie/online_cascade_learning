from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from utils import *

global DATASET
DATASET = 'hatespeech'

SytemPrompt = "You are given a post from an online forum and you need to check whether the post contains any hate speech. Return your answer in one word (yes or no) without any explanations. "
Prompt = '''{}'''
PROMPT = " ".join(["[INST]", B_SYS, SytemPrompt, E_SYS, Prompt, "[/INST]"])

def postprocess(output):
    if "no" in output:
        return 0
    elif "yes" in output:
        return 1
    else:
        return 0