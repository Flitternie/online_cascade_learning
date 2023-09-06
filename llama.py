import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def inference(dataloader, temperature=0.3, max_new_tokens=100, top_k=5):
    # load model
    model_card = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_card) 
    model = model.to('cuda')
    print("Model loaded")

    outputs = []
    original_outputs = []
    print("Starting inference...")
    for batch in tqdm(dataloader):
        batch = tokenizer.batch_encode_plus(batch, padding=True, return_tensors='pt')
        batch = batch.to('cuda')
        input_length = batch['input_ids'].shape[1]
        output = model.generate(
            **batch, 
            do_sample=True, 
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
        output = tokenizer.decode(output[0][input_length:], skip_special_tokens=True).lower().strip()
        original_outputs.append(output.replace("\n", " "))
    
    return original_outputs