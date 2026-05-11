from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    def __init__(self):
        model_name = "facebook/bart-large-cnn"
        #model_name = "miocrosoft/Phi-3-mini-4k-instruct"
        print(f"[Generator] Loading generator model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print("[Generator] Generator model loaded")

    def generate(self, query, context):
        """Generate an answer using Flan-T5 given a query and retrieved context."""
        prompt = (
            f"You are a helpful medical assistant. "
            f"Answer the following question based on the provided context.\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            num_beams=2,
            early_stopping=True
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the answer
        answer = " ".join(answer.strip().split())

        if len(answer.split()) < 3:
            return "I don't have enough information to answer that question."

        return answer
