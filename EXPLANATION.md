# Project Explanation (Files and Connections)

This document explains the responsibility of each file and how they connect to form the RAG pipeline.

## High-Level Flow

1. `main.py` loads configuration and environment variables.
2. Documents are loaded and chunked in `src/document_ingestion/__init__.py`.
3. Embeddings + vector store are built in `src/vectorstore/__init__.py`.
4. LangGraph nodes are defined in `src/nodes/__init__.py`.
5. The graph is assembled in `src/graph_builder/__init__.py`.
6. User questions are passed through the graph and answered via the LLM.

## File-by-File Responsibilities

### `main.py`
- Entry point for the CLI.
- Loads `.env` (via `python-dotenv`) and reads configuration defaults.
- Builds the vector store and LangGraph pipeline.
- Handles user input loop and prints answers.
- Optional `--visualize` renders the graph using matplotlib + networkx.

### `src/config/__init__.py`
- Centralized defaults: chunk sizes, models, temperature, retrieval k.
- `CUSTOM_PROMPT` template used by the generate node.
- This isolates configuration changes from logic changes.

### `src/state/__init__.py`
- Defines the LangGraph `State` TypedDict.
- State carries `question`, retrieved `context`, and final `answer`.
- Ensures each node knows the expected keys.

### `src/document_ingestion/__init__.py`
- `load_json_documents`: reads `knowledge_base.json` (list of `{ "text": ... }`).
- `split_documents`: chunks documents into overlapping slices for retrieval.
- This is the only place that touches raw source data.

### `src/vectorstore/__init__.py`
- `build_vector_store`: creates embeddings and stores them in memory.
- Uses OpenAI embeddings + `InMemoryVectorStore`.
- This is the retrieval index for similarity search.

### `src/nodes/__init__.py`
- `classify`: placeholder for routing/metadata logic (currently no-op).
- `make_retrieve_node`: searches the vector store for relevant chunks.
- `make_generate_node`: formats prompt and calls the LLM.
- `refine`: appends a small refinement note to the answer.
- These functions are the LangGraph execution steps.

### `src/graph_builder/__init__.py`
- `build_graph`: wires `classify → retrieve → generate → refine`.
- Returns both the compiled graph and the builder (for visualization).
- `visualize_langgraph_clean`: draws the workflow diagram.

## Connection Map

```
main.py
  ├─ loads settings from src/config/__init__.py
  ├─ loads/splits docs via src/document_ingestion/__init__.py
  ├─ builds vector store via src/vectorstore/__init__.py
  ├─ builds graph via src/graph_builder/__init__.py
  │    └─ uses node functions from src/nodes/__init__.py
  └─ invokes graph with State defined in src/state/__init__.py
```

## Design Notes

- Each module has a single responsibility so you can swap pieces later.
- The graph is explicit, so adding steps (rerank, guardrails, tools) is easy.
- The `knowledge_base.json` format is deliberately simple for early iteration.
