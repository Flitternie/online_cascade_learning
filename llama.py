import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaForSequenceClassification
from tqdm import tqdm

class LlamaModel():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(args.model) 
        self.model = self.model.to('cuda')
        print("Llama Model loaded")

    def inference(self, dataloader):
        outputs = []
        original_outputs = []
        print("Starting inference...")
        for batch in tqdm(dataloader):
            batch = self.tokenizer.batch_encode_plus(batch, padding=True, return_tensors='pt')
            batch = batch.to('cuda')
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

    def predict(self, input):
        input = self.tokenizer.encode_plus(input, return_tensors='pt').to('cuda')
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