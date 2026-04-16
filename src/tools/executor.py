# src/tools/executor.py

from src.retriever.search import Retriever

retriever = Retriever()


def execute_tool(tool_name, args):
    if tool_name == "SearchKB":
        return retriever.search(args["query"], k=3)

    elif tool_name == "CreateTicket":
        return {
            "status": "ticket_created",
            "issue": args["issue"]
        }

    else:
        return {"error": "Unknown tool"}