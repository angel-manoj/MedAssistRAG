from src.rag_pipeline import RAG

rag = RAG()

while True:
    query = input("\nAsk medical question: ")

    answer, docs = rag.ask(query)

    print("\n--- RETRIEVED CONTEXT ---")
    for d in docs:
        print("-", d[:150])

    print("\n--- ANSWER ---")
    print(answer)