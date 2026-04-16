import ollama


class LLM:
    def __init__(self):
        self.model = "llama3"   # or "phi3" for faster

    def generate(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]