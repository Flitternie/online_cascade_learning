import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import tiktoken

# Load dataset from csv file
isear_dataset = pd.read_csv('./data/isear_preprocessed.csv', sep=',', header=0, names=['label', 'text'])

SystemPrompt = '''
In this task, you will be performing a classification exercise aimed at identifying the underlying emotion conveyed by a given sentence.
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
Remember, you have to classify the sentences using only anger, joy, sadness, guilt, shame, fear or disgust. Please respond with only the word and nothing else.
'''

UserPrompt = '''
Sentence: {} 
Classify the emotion of this hypothetical sentence. Respond in exactly one word in all lowercase with a response in the exact format requested by the user. 
'''

# Read API key from file ~/OPENAI_KEY
openai_api_key = open(os.path.expanduser("~/OPENAI_KEY")).read().strip()
openai_org_id = open(os.path.expanduser("~/OPENAI_ORG")).read().strip()

client = OpenAI(
  api_key=openai_api_key,
  organization=openai_org_id,
)

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")
enc.encode("anger"), enc.encode("joy"), enc.encode("sadness"), enc.encode("guilt"), enc.encode("shame"), enc.encode("fear"), enc.encode("disgust")

isear_logit_bias = {
  4091: 100,
  4215: 100,
  83214: 100,
  2136: 100,
  8890: 100,
  3036: 100,
  939: 100,
  373: 100,
  69: 100,
  686: 100,
  4338: 100,
  70: 100,
  592: 100,
  100257: 100,
}


# Convert logprobs to a dictionary
def log_probs_to_dict(logprobs):
  token_dict = {}
  for prob in logprobs:
    token_dict[prob.token] = prob.logprob
  return token_dict


f = open("./gpt_results/gpt3.5/isear_gpt3.5_turbo_1106.txt", "w", buffering=1)
g = open("./gpt_results/gpt3.5/isear_gpt3.5_turbo_1106_probs.txt", "w", buffering=1)

pbar = tqdm(total=len(isear_dataset['text']))

correct, total = 0, 0
for i in range(len(isear_dataset['text'])):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    seed=42,
    logprobs=True,
    top_logprobs=5,
    logit_bias=isear_logit_bias,
    temperature=0.3,
    max_tokens=5,
    messages=[
      {"role": "system", "content": SystemPrompt},
      {"role": "user", "content": UserPrompt.format(isear_dataset['text'][i])},
    ]
  )
  text_output = response.choices[0].message.content
  f.write(text_output + "\n")
  g.write(str(log_probs_to_dict(response.choices[0].logprobs.content[0].top_logprobs)) + "\n")

  correct += (text_output.lower().strip() == isear_dataset['label'][i])
  total += 1
  pbar.set_description("Accuracy: {}".format(correct/total))
  pbar.update(1)

f.close()
g.close()