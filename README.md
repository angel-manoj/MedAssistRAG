# DL Healthcare — RAG-based Medical Assistant

This repository contains a retrieval-augmented generation (RAG) pipeline focused on medical question answering. It combines a document retriever (FAISS + SentenceTransformers), an optional reranker, and a language model component (either via Ollama or local Transformers + adapters) to produce evidence-backed answers.

## High-level overview

- The pipeline retrieves relevant medical passages from a pre-built FAISS index using a SentenceTransformers embedding model.
- Retrieved passages are optionally re-ranked using a fine-tuned reranker to improve top-k precision.
- A generator (LLM) consumes the top passages as context and produces a final answer. The generator can be accessed via:
  - Ollama chat (recommended if you run Ollama locally and prefer a hosted model), or
  - Local Transformers models with PEFT adapters (SFT + DPO) if you have the model artifacts in `models/` and sufficient system resources.
- There is also a tool-execution layer that can decide whether to search the KB, create a ticket, or add a medical disclaimer to the response.

## Repository structure (important files/folders)

- `main.py` — Simple interactive entrypoint that constructs `src.rag_pipeline.RAG` and prompts the user for questions.
- `src/rag_pipeline.py` — Tool-enabled RAG pipeline. Decides which tool to run, executes it, and generates final answers.
- `src/generator/llm.py` — LLM wrapper. By default this file uses `ollama.chat` to call a local Ollama model. The file also contains (commented) code for a local Transformers + PEFT setup.
- `src/retriever/run_pipeline.py` — Script to build embeddings and create/save a FAISS index from processed data.
- `src/retriever/search.py` — Retriever class that loads FAISS, the embedding model, and (optionally) reranker to return top passages.
- `src/tools/` — Tool implementations and `executor.py` to call them from `rag_pipeline`.
- `data/processed/` — Processed datasets and artifacts used to create embeddings and indexes (e.g., `qa_dataset_large.csv`).
- `models/` — Saved model artifacts, including fine-tuned retriever, generator PEFT adapters, and other checkpoints. Large files are tracked outside git.

## Quick contract (what the code expects)

- Inputs: user natural-language question (string). The retriever expects the FAISS index and texts saved in `data/embeddings/`.
- Outputs: an answer string and a list of source passages used to form the answer.
- Error modes: missing model/index files will raise file I/O errors; Ollama must be running locally if used; local transformer execution requires GPU/CPU resources and matching model checkpoints.

## How the pipeline flows (detailed)

1. Tool Decision (LLM): `RAGPipeline.query()` calls `LLM.generate_raw()` with a small prompt asking which tool to use. The LLM must reply with JSON like `{"tool": "SearchKB", "args": {"query": "..."}}`.
2. Tool Execution: `execute_tool()` from `src/tools/executor.py` runs the chosen tool. The main tools are:
   - `SearchKB(query)` — run retriever search and return a list of passages
   - `CreateTicket(issue)` — create a support ticket (implementation project-specific)
   - `MedicalDisclaimerTool(query)` — returns a disclaimer and may rely on `SearchKB` internally
3. Answer Generation: If the tool is `SearchKB`, the top passages are joined into a context string and passed to `LLM.generate(question, context)` to produce the final answer.

If the tool is `MedicalDisclaimerTool`, the pipeline searches the KB, generates a base answer, then appends the disclaimer returned by the tool.

## Dependencies

Primary Python dependencies are listed in `requirements.txt`. Key packages:

- torch, transformers, peft, trl — for local LLM usage
- sentence-transformers, faiss-cpu — for embeddings + retrieval
- pandas, numpy, scikit-learn — data processing
- streamlit — optional UI

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If you plan to use the Ollama integration (default in `src/generator/llm.py`), install and run Ollama separately: https://ollama.com/
- If you want to run the local Transformers-based generator you must have the base model and adapters in `models/` and uncomment the alternative code in `src/generator/llm.py`.

## Setup steps (recommended order)

1. Prepare data
   - Ensure processed dataset exists: `data/processed/qa_dataset_large.csv` (the retriever script uses this by default).
2. Build embeddings & FAISS index
   - Run the retriever pipeline which creates embeddings and saves the FAISS index and texts:

```bash
python src/retriever/run_pipeline.py
```

3. Start Ollama (optional)
   - If using Ollama-based LLM, start the Ollama daemon on your machine and ensure the model name in `src/generator/llm.py` (default `llama3`) is available.

4. Run the interactive RAG shell

```bash
python main.py
```

You can then type medical questions and receive answers with printed sources.

## Using local Transformers + PEFT generator (optional)

If you want to use the local generator instead of Ollama, follow these extra steps:

1. Place base model and PEFT adapters in `models/`:
   - Base model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or other) should be accessible by the tokenizer and model loader.
   - SFT adapter in `models/generator_finetuned`
   - DPO adapter in `models/dpo_aligned_model`

2. Edit `src/generator/llm.py` to enable the Transformers-based LLM (there is commented code present). Then run `main.py`.

Caveats:
- Local model loading may require significant RAM and/or a GPU. The code defaults to CPU in the commented example but will be very slow without GPU.

## Common tasks & commands

- Rebuild index after data changes:

```bash
python src/retriever/run_pipeline.py
```

- Test retriever alone (small script):

```python
from src.retriever.search import Retriever
r = Retriever()
print(r.search("what is hypertension?")[:3])
```

- Test RAG pipeline with Ollama (interactive):

```bash
python main.py
```

## Troubleshooting

- Ollama errors: ensure Ollama is installed and the named model is present. Check `ollama` CLI and logs.
- FAISS index not found: run `src/retriever/run_pipeline.py` to (re)create the index. The index file expected is under `data/embeddings/` (check `src/retriever/search.py` for exact filenames).
- Missing model files: verify paths inside `models/` match the ones referenced in code (`models/retriever_finetuned`, `models/generator_finetuned`, `models/dpo_aligned_model`).

## Next steps & suggestions

- Add a small CLI wrapper for common operations (`build-index`, `serve`, `interactive`) to simplify usage.
- Add unit tests for retriever and RAG pipeline (happy path + missing-files edge cases).
- Add a small example notebook demonstrating a full example query and outputs.

## Acknowledgements

This project re-uses common open-source libraries for RAG-style systems (FAISS, SentenceTransformers, Hugging Face Transformers, PEFT). Keep track of license requirements for any models you download and use.

---

If you'd like, I can also:

- Generate example commands for running the Ollama model locally and matching the `llm.py` usage.
- Add a basic CLI (`scripts/cli.py`) that exposes index-building and interactive modes.
