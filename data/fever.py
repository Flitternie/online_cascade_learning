from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from datasets import Dataset

DATASET = 'fever'

SystemPrompt = '''You are a helpful, respectful and honest assistant. 
                  This is a fact-checking task. You are to use your knowledge to determine whether a given claim is true or false.
                  Only answer either ’true’ or ’false’.'''
Prompt = '''The following claim was made: {} \n Was this claim true or false? Return the answer in one word.'''
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, Prompt, "[/INST]"])

def preprocess(data: Dataset) -> Dataset:
    # change label REFUTES to 0 and label SUPPORTS to 1
    data = data.map(lambda example: {'label': 0 if example['label'] == 'REFUTES' else 1, 'text': example['text']})
    return data

def postprocess(output: str) -> int:
    output = output.lower().strip()
    try:
        return int(output)
    except ValueError:
        if "true" in output:
            return 1
        elif "false" in output:
            return 0
        else:
            return -1
    
def _postprocess(output: str) -> int:
    output = output.lower().strip()
    if "support" in output:
        return 1
    elif "refute" in output:
        return 0
    else:
        return -1