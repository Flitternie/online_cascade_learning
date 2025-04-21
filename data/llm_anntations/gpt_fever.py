import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import tiktoken

# Load dataset from csv file
fever_dataset = pd.read_csv('./data/fever_preprocessed.csv', sep=',', header=0, names=['label', 'text'])
fever_label_converter = {"SUPPORTS": "true", "REFUTES": "false"}

SystemPrompt = '''
You are a helpful, respectful and honest assistant. This is a fact-checking task. 
Use your knowledge to determine whether a given claim is true or false. 
Answer only in "true" or "false" without providing any explanations. 
'''

UserPrompt = 'On June 2017, the following claim was made: {} '

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
enc.encode("true"), enc.encode("false")

fever_logit_bias = {
  1904: 100,
  3934: 100,
  100257: 100,
}


# Convert logprobs to a dictionary
def log_probs_to_dict(logprobs):
  token_dict = {}
  for prob in logprobs:
    token_dict[prob.token] = prob.logprob
  return token_dict


f = open("./gpt_results/gpt3.5/fever_gpt3.5_turbo_1106.txt", "w", buffering=1)
g = open("./gpt_results/gpt3.5/fever_gpt3.5_turbo_1106_probs.txt", "w", buffering=1)

pbar = tqdm(total=len(fever_dataset['text']))

correct, total = 0, 0
for i in range(len(fever_dataset['text'])):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    seed=42,
    logprobs=True,
    top_logprobs=5,
    logit_bias=fever_logit_bias,
    temperature=0.3,
    max_tokens=5,
    messages=[
      {"role": "system", "content": SystemPrompt},
      {"role": "user", "content": UserPrompt.format(fever_dataset['text'][i])},
    ]
  )
  text_output = response.choices[0].message.content
  f.write(text_output + "\n")
  g.write(str(log_probs_to_dict(response.choices[0].logprobs.content[0].top_logprobs)) + "\n")

  correct += (text_output.lower().strip() == fever_label_converter[fever_dataset['label'][i]])
  total += 1
  pbar.set_description("Accuracy: {}".format(correct/total))
  pbar.update(1)

f.close()
g.close()