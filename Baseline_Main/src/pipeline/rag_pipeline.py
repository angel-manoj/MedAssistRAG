from src.retriever.retriever import Retriever
from src.generator.generator import Generator


class RAGPipeline:
    def __init__(self):
        print("\n[RAG] Initializing Baseline RAG Pipeline...")
        self.retriever = Retriever()
        self.generator = Generator()
        print("[RAG] Pipeline ready!\n")

    def query(self, question):
        """Run the full RAG pipeline: retrieve context then generate answer."""

        # Step 1: Retrieve relevant QA pairs
        results = self.retriever.search(question, top_k=5)

        if results.empty:
            return {
                "question": question,
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }

        # Step 2: Build context from retrieved answers
        context = "\n\n".join(results["answer"].tolist())

        # Step 3: Generate answer using Flan-T5
        answer = self.generator.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": results["question"].tolist()
        }
