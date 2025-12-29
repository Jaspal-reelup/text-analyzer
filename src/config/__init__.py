DEFAULT_KNOWLEDGE_BASE_PATH = "knowledge_base.json"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 5

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_LLM_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.3

CUSTOM_PROMPT = (
    "You are an advanced assistant. Use the context to answer. "
    "If insufficient info, say so clearly.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n"
)
