import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from tqdm import tqdm
from utils import *

class LlamaModel():
    def __init__(self, args: ModelArguments):
        self.args = args
        self.device = args.device
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(args.model, device_map="auto")
        print("Llama Model loaded")

    def inference(self, dataloader: DataLoader) -> list[str]:
        outputs = []
        original_outputs = []
        print("Starting inference...")
        for batch in tqdm(dataloader):
            batch = self.tokenizer.batch_encode_plus(batch[0], padding=True, truncation=True, return_tensors='pt')
            batch = batch.to(self.device)
            input_length = batch['input_ids'].shape[1]
            output = self.model.generate(
                **batch, 
                do_sample=True, 
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_new_tokens,
                top_k=self.args.top_k,
            )
            output = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).lower().strip()
            original_outputs.append(output.replace("\n", " "))
        
        return original_outputs

    def predict(self, input: str) -> tuple[str, float]:
        input = self.tokenizer.encode_plus(input, return_tensors='pt').to(self.device)
        input_length = input['input_ids'].shape[1]
        output = self.model.generate(
            **input, 
            do_sample=True, 
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_new_tokens,
            top_k=self.args.top_k,
        )
        output = self.tokenizer.decode(output[0][input_length:], skip_special_tokens=True).lower().strip()
        torch.cuda.empty_cache()
        return output.replace("\n", " "), 1.0