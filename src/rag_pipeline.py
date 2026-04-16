from src.generator.llm import LLM
from src.tools.executor import execute_tool
import json


def tool_prompt(query):
    return f"""
You are an AI assistant with access to tools.

Available tools:
1. SearchKB(query) - search knowledge base
2. CreateTicket(issue) - create support ticket

Decide the best tool.

Return ONLY JSON:
{{"tool": "...", "args": {{...}}}}

Query: {query}
"""


def build_final_prompt(context, query):
    return f"""
You are a medical assistant.

Answer using ONLY the context below.
Give a clear 2-3 sentence answer.

Context:
{context}

Question:
{query}

Answer:
"""


def main():
    llm = LLM()

    print("🧠 Tool-enabled RAG ready (type 'exit')\n")

    while True:
        query = input("Ask question: ")

        if query.lower() in ["exit", "quit"]:
            break

        # 🧠 STEP 1: Decide tool
        decision_raw = llm.generate(tool_prompt(query))

        try:
            decision = json.loads(decision_raw)
        except:
            print("⚠️ Failed to parse tool decision, using fallback")
            decision = {"tool": "SearchKB", "args": {"query": query}}

        print("\n🛠 Tool Decision:", decision)

        # ⚙️ STEP 2: Execute tool
        result = execute_tool(decision["tool"], decision["args"])

        print("\n📄 Tool Result:", result)

        # 🧠 STEP 3: Generate final answer
        if decision["tool"] == "SearchKB":
            context = "\n".join(result)
            final_prompt = build_final_prompt(context, query)
            answer = llm.generate(final_prompt)

        else:
            answer = str(result)

        print("\n💡 Answer:\n", answer)
        print("-" * 50)


if __name__ == "__main__":
    main()