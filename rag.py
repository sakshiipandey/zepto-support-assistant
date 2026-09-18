
import os
import time
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load embedding model once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB using absolute path
CHROMA_PATH = os.path.join(BASE_DIR, "vector_store")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="support_docs")

# Kie.ai / LLM client configuration
api_key = os.getenv("KIE_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.kie.ai/gpt-5-2/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-2")
MOCK_LLM = os.getenv("MOCK_LLM", "false").lower() in ("true", "1", "yes")

if not api_key:
    logger.warning("Neither KIE_API_KEY nor OPENAI_API_KEY is set in environment/.env")

# Avoid crashing on import if api_key is None by providing a fallback string
llm = OpenAI(
    api_key=api_key or "missing-key",
    base_url=OPENAI_BASE_URL,
)


def _build_prompt(context: str, question: str) -> str:
    """Build the system prompt for policy-based answers."""
    return f"""ROLE:
You are a helpful Zepto customer support assistant.

CONTEXT:
{context}

TASK:
Answer the user's question using ONLY the context.

RULES:
- Do not invent information.
- If the answer is not in the context, say that the provided policy information does not contain the answer.
- Keep the answer short and clear.

EXAMPLE:

Question:
What should I do if my order contains a damaged item?

Answer:
You should report the damaged item according to the damaged item policy.

USER QUESTION:
{question}
"""


def _call_llm(prompt: str, context: str = "") -> str:
    """Send prompt to Kie.ai LLM with comprehensive error handling and fallback."""
    # If mock mode is enabled, return a synthesized response without making network calls
    if MOCK_LLM:
        logger.info("Mock LLM enabled — synthesizing response without API call")
        if context:
            clean_ctx = "\n".join([line for line in context.split("\n") if line.strip() and not line.startswith("SAMPLE POLICY")])
            return f"Based on Zepto policy:\n{clean_ctx}"
        return "I am a Zepto support assistant. How can I assist you with your order today?"

    if not api_key:
        return (
            "Kie.ai API key is missing. Please set KIE_API_KEY (or OPENAI_API_KEY) "
            "in your .env file to enable live AI responses."
        )

    start = time.time()
    try:
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        logger.info("Kie.ai LLM time: %.2fs", time.time() - start)

        # Kie.ai can return HTTP 200 with an error object, e.g. {"code": 401, "msg": "..."}
        # In this scenario, OpenAI SDK parses it into ChatCompletion with choices=None
        code = getattr(response, "code", None)
        if code and code != 200:
            msg = getattr(response, "msg", "Authentication or quota error")
            logger.error("Kie.ai returned error code %s: %s", code, msg)
            return f"Kie.ai API Error ({code}): {msg}. Please check your API key in .env."

        choices = getattr(response, "choices", None)
        if not choices:
            msg = getattr(response, "msg", None) or "Model did not return choices"
            logger.error("Kie.ai response missing choices: %s", response)
            return f"Kie.ai did not return an answer ({msg})."

        content = choices[0].message.content
        if not content:
            return "The assistant could not generate an answer."

        return content.strip()

    except Exception as exc:
        logger.error("Failed to query Kie.ai LLM: %s", exc, exc_info=True)
        return f"Error connecting to Kie.ai: {str(exc)}"


def answer_question(question: str) -> str:
    """Retrieve relevant chunks from ChromaDB and generate an answer."""
    total_start = time.time()

    # Embed the question
    start = time.time()
    question_embedding = model.encode([question], show_progress_bar=False)
    logger.info("Embedding time: %.2fs", time.time() - start)

    # Search similar chunks
    start = time.time()
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3,
    )
    logger.info("ChromaDB time: %.2fs", time.time() - start)

    docs = results.get("documents", [[]])
    context = "\n\n".join(docs[0]) if docs and docs[0] else ""

    # Generate answer via LLM
    prompt = _build_prompt(context, question)
    answer = _call_llm(prompt, context=context)
    logger.info("Total time: %.2fs", time.time() - total_start)

    return answer


def direct_answer(question: str) -> str:
    """Answer general questions that don't need RAG retrieval."""
    prompt = f"""You are a helpful customer support assistant.

Answer this general question politely and briefly.
Do not invent Zepto policy information.

Question:
{question}
"""
    return _call_llm(prompt)

