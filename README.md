# text-analyzer

Brief note: This is a minimal Retrieval-Augmented Generation (RAG) demo using LangChain for ingestion, embeddings, and LLM calls, and LangGraph for orchestration. It loads a JSON knowledge base, chunks and embeds it, retrieves relevant context for user questions, and generates concise answers.

## Setup

1. Create a `knowledge_base.json` in the project root:

```json
[
  { "text": "Document chunk one." },
  { "text": "Document chunk two." }
]
```

2. Export your OpenAI key:

```bash
export OPENAI_API_KEY="your-key"
```

3. Run:

```bash
python main.py
```

Optional visualization (requires `networkx` and `matplotlib`):

```bash
python main.py --visualize
```

## Large PDF Corpus Mode

For many PDFs, build a persistent FAISS index once and reuse it:

```bash
python main.py --use-pdfs --pdf-dir data/pdfs --persist-dir data/vectorstore --rebuild-index
```

Subsequent runs can load the saved index:

```bash
python main.py --use-pdfs --pdf-dir data/pdfs --persist-dir data/vectorstore
```
