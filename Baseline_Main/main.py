from src.pipeline.rag_pipeline import RAGPipeline


if __name__ == "__main__":
    print("=" * 50)
    print("  Baseline Healthcare RAG ChatBot")
    print("=" * 50)

    # Initialize the RAG pipeline
    rag = RAGPipeline()

    print("Type your medical question below (or 'exit' to quit).\n")

    while True:
        query = input("You: ")

        if query.lower().strip() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        if not query.strip():
            continue

        result = rag.query(query)

        print(f"\nAnswer: {result['answer']}")

        if result["sources"]:
            print("\nRetrieved from:")
            for i, src in enumerate(result["sources"][:3], 1):
                print(f"   {i}. {src}")

        print()
