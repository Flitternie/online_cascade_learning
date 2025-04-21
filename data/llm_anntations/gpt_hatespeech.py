import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import tiktoken
from data.hatespeech import postprocess

# Load dataset
hatespeech_dataset = pd.read_csv('./data/hatespeech_preprocessed.csv', sep=',', header=0, names=['label', 'text'])

SystemPrompt = "You are given a post from an online forum and you need to check whether the post contains any hate speech. Return your answer in one word (yes or no) without any explanations."
UserPrompt = '''Post: {}'''

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
enc.encode("yes"), enc.encode("no")

hatespeech_logit_bias = {
  9891: 100,
  2201: 100,
  100257: 100,
}


# Convert logprobs to a dictionary
def log_probs_to_dict(logprobs):
  token_dict = {}
  for prob in logprobs:
    token_dict[prob.token] = prob.logprob
  return token_dict


f = open("./gpt_results/gpt3.5/hatespeech_gpt3.5_turbo_1106.txt", "w", buffering=1)
g = open("./gpt_results/gpt3.5/hatespeech_gpt3.5_turbo_1106_probs.txt", "w", buffering=1)

pbar = tqdm(total=len(hatespeech_dataset['train']['text']))

correct, total = 0, 0
positive_correct, positive_total = 0, 0
for i in range(len(hatespeech_dataset['train']['text'])):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    seed=42,
    logprobs=True,
    top_logprobs=5,
    logit_bias=hatespeech_logit_bias,
    temperature=0.3,
    max_tokens=5,
    messages=[
      {"role": "system", "content": SystemPrompt},
      {"role": "user", "content": UserPrompt.format(hatespeech_dataset['train'][i]['text'])},
    ]
  )
  text_output = response.choices[0].message.content
  prediction = postprocess(text_output)
  f.write(text_output + "\n")
  g.write(str(log_probs_to_dict(response.choices[0].logprobs.content[0].top_logprobs)) + "\n")

  correct += (prediction == hatespeech_dataset['train'][i]['label'])
  total += 1
  if hatespeech_dataset['train'][i]['label'] == 1:
    positive_correct += (prediction == hatespeech_dataset['train'][i]['label'])
    positive_total += 1

  pbar.set_description("Accuracy: {:.4f}, Positive Acc: {:.4f}".format(correct/total, positive_correct/max(1, positive_total)))
  pbar.update(1)

f.close()
g.close()