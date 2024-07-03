# Data Module Specification

This document specifies the general requirements and guidelines for implementing a data module for different datasets. Ensure that each component is properly defined and formatted to maintain readability and functionality.

## General Structure

A data module implementation should include the following components:

1. **Imports**
2. **Dataset Definition**
3. **System Prompt Definition**
4. **User Prompt Definition**
5. **Prompt Construction**
6. **Label List Definition**
7. **Preprocess Function**
8. **Postprocess Function**

## Components

### 1. Imports
Necessary libraries and modules should be imported at the beginning of the file. Common imports include:
```python
from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from datasets import Dataset
```

### 2. Dataset Definition
Define the dataset name as a constant string.
```python
DATASET = 'dataset_name'
```

### 3. System Prompt Definition
Define a system prompt that explains the task to the model. The prompt should be formatted as a string.
```python
SystemPrompt = "Your task description here."
```

### 4. User Prompt Definition
Define a user prompt that includes placeholders for the input data. The prompt should be formatted as a string.
```python
UserPrompt = '''Your input template here: {} \n Additional instructions.'''
```

### 5. Prompt Construction
Construct the final prompt using the system and user prompts along with any special tokens.
```python
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, UserPrompt, "[/INST]"])
```

### 6. Label List Definition
Define a list of possible labels for the classification task.
```python
LabelList = ['label1', 'label2', 'label3', ...]
```

### 7. Preprocess Function
Define a `preprocess` function that takes a dataset and returns a preprocessed dataset. This function is responsible for preparing the data for training or evaluation. The preprocessed dataset should have a `text` column containing the input text and a `label` column containing the corresponding label.
```python
def preprocess(data: Dataset) -> Dataset:
    # Preprocessing logic here
    return data
```

### 8. Postprocess Function
Define a `postprocess` function that takes the LLM output as a string and returns the corresponding label as an integer.
```python
def postprocess(output: str) -> int:
    # Postprocessing logic here
    return label_id
```

## Example Implementations
Check [`imdb.py`](./imdb.py), [`hatespeech.py`](./hatespeech.py), [`isear.py`](./isear.py), [`fever.py`](./fever.py) for example implementations of the data module.