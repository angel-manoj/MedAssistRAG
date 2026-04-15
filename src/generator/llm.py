from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLM:
    def __init__(self):
        model_name = "microsoft/phi-2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

    def generate(self, prompt, max_new_tokens=200):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)