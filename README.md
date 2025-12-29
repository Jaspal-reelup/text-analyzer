# text-analyzer

Minimal RAG system wired with LangChain + LangGraph.

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
