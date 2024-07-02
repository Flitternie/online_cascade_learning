from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
from datasets import Dataset

DATASET = 'isear'

SystemPrompt = """In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence. 
 The emotions to consider are as follows: 
 Anger: Anger is a strong feeling of displeasure, hostility, or frustration.  
 Joy: Joy is a positive and uplifting emotion characterized by happiness, elation, and a sense of contentment.  
 Sadness: Sadness is a feeling of sorrow, unhappiness, or despondency. 
 Guilt: Guilt is a self-directed emotion that arises from a sense of wrongdoing or moral transgression.  
 Shame: Shame is a powerful emotion associated with feeling embarrassed, humiliated, or unworthy.  
 Fear: Fear is an emotion triggered by a perceived threat or danger.  
 Disgust: Disgust is an aversive emotion linked to feelings of revulsion, repulsion, or strong distaste. It arises in response to things that are offensive or unpleasant. 
 Your task is to analyze each sentence provided and categorize it into one of these emotions based on the dominant feeling conveyed by the text. 
 This classification will require an understanding of the nuances of human emotions and the context in which the sentences are presented. 
 Remember, you have to classify the sentences using only anger, joy, sadnes, guilt, shame, fear or disgust. Please respond with only the word and nothing else.  
 """
UserPrompt = '''{} \n Classify the emotion of this hypothetical sentence. Respond in exactly one word in all lowercase with a response in the exact format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer. Only answer in exactly one word.'''
PROMPT = " ".join(["[INST]", B_SYS, SystemPrompt, E_SYS, UserPrompt, "[/INST]"])

LabelList = ['joy', 'sadness', 'anger', 'guilt', 'shame', 'fear', 'disgust']

isear_to_id = {
  "joy":      0,
  "sadness":  1,
  "anger":    2,
  "guilt":    3,
  "shame":    4,
  "fear":     5,  
  "disgust":  6,
}

def preprocess(data: Dataset) -> Dataset:
  data = data.map(lambda e: {'label': isear_to_id[e['label']]})
  return data

def postprocess(output: str) -> int:
  low_output = output.lower().strip()
  try:
    return int(low_output)
  except ValueError:
    for k, v in isear_to_id.items():
      if k in low_output:
        return v
    return -1