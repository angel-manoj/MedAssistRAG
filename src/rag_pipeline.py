from src.retriever.search import Retriever
from src.generator.llm import LLM

class RAG:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLM()

    def build_prompt(self, query, docs):
        context = "\n\n".join(docs)

        prompt = f"""
You are a medical assistant. Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt

    def ask(self, query):
        docs = self.retriever.search(query, k=5)

        prompt = self.build_prompt(query, docs)

        answer = self.llm.generate(prompt)

        return answer, docs