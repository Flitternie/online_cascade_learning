from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from utils import *

DATASET = 'fever'

SystemPrompt = '''You are a helpful, respectful and honest assistant. 
                  This is a fact-checking task. You are to use your knowledge to determine whether a given claim is true or false.
                  Only answer either ’true’ or ’false’.'''
Prompt = '''The following claim was made: {} \n Was this claim true or false? Return the answer in one word.'''
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, Prompt, "[/INST]"])

def postprocess(output):
    output = output.lower().strip()
    if "true" in output:
        return 1
    elif "false" in output:
        return 0
    else:
        return -1
    
def postprocess2(output):
    output = output.lower().strip()
    if "support" in output:
        return 1
    elif "refute" in output:
        return 0
    else:
        return -1