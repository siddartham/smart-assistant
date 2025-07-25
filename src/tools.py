from dotenv import load_dotenv
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from .structured_data_simulator import structured_data
from .indexer import embedding_model
from .logger import PromptLoggingCallbackHandler, logger

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[PromptLoggingCallbackHandler()])
vectordb = FAISS.load_local("vectorstore/index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever()


def rerank_with_cross_encoder(query, docs, top_k=2):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    logger.info("[Cross-Encoder Reranking Results]:")
    for i, (doc, score) in enumerate(scored_docs[:top_k]):
        preview = doc.page_content.replace("\n", " ")[:200]
        logger.info(f"Rank {i+1}: Score={score:.4f} | Preview: {preview}...")

    return [doc for doc, _ in scored_docs[:top_k]]


def document_qa_tool(query: str) -> dict:
    logger.info(f"[Tool: Document Search] Query: {query}")

    initial_docs = retriever.get_relevant_documents(query) # Top 10

    top_docs = rerank_with_cross_encoder(query, initial_docs, top_k=2)  # rerank to top-2
    context = "\n\n".join(doc.page_content for doc in top_docs)
    qa_prompt = f"Use the context below to answer this question: {query}\n\nContext:\n{context}"
    answer = llm.invoke(qa_prompt).content
    logger.info(f"[Document Search Answer]: {answer}")

    reflection_prompt = (
        f"User asked: '{query}'\n"
        f"Document answer: '{answer}'\n\n"
        "Does this answer confidently and fully address the question using the document context? "
        "If yes, respond with 'document'. If not, respond with 'structured'.\n"
        "Consider escalating to structured data if:\n"
        "- Numeric comparisons or trends are unclear\n"
        "- Answer is vague or incomplete\n"
        "- The document context lacks specific figures or tables"

    )

    escalation_decision = llm.invoke(reflection_prompt).content.strip().lower()

    if escalation_decision == "structured":
        logger.info("[Tool Decision] Escalating to structured data based on evaluation.")
        structured_result = query_structured_data(query, "document_search")
        return {
            "answer": structured_result["answer"],
            "tool": "structured_data_lookup",
            "tool_trace": ["document_search", "structured_data_lookup"]
        }

    return {
        "answer": answer,
        "tool": "document_search",
        "tool_trace": ["document_search"]
    }


def summarize_doc(query: str) -> dict:

    logger.info(f"[Tool: Summarization] Query: {query}")
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    summary_prompt = PromptTemplate.from_template("""
        You are a report summarizer.

        User question: "{query}"

        Based on the following document content, write a high-level, concise, and factual summary.
        Avoid guessing. Use only the provided content.

        --- Document Section Start ---
        {context}
        --- Document Section End ---

        Respond in bullet points with one line each:
        - Insight 1
        - Insight 2
        - ...
        """)
    summary_chain = summary_prompt | llm
    summary = summary_chain.invoke({"query": query, "context": context})
    logger.info(f"[Document Summary] Answer: {summary}")

    reflection_prompt = (
        f"User asked: '{query}'\n"
        f"Summary output: '{summary}'\n\n"
        "Does this summary fully address the question using the document context? "
        "If not, respond with 'structured'. Otherwise respond with 'document'."
    )
    escalation_decision = llm.invoke(reflection_prompt).content.strip().lower()

    if escalation_decision == "structured":
        logger.info("[Tool Decision] Escalating to structured data based on summary evaluation.")
        structured_result = query_structured_data(query, "summarize_doc")
        return {
            "answer": structured_result["answer"],
            "tool": "structured_data_lookup",
            "tool_trace": ["summarization_tool", "structured_data_lookup"]
        }

    return {
        "answer": summary,
        "tool": "summarization_tool",
        "tool_trace": ["summarization_tool"]
    }


def query_structured_data(query: str, source_tool: str) -> dict:
    logger.info(f"[Tool: Structured Data] Query: {query}")

    # Construct prompt to translate natural query into pandas code
    prompt = f"""
You are a data assistant. A user asked: "{query}".

You are provided with a Pandas DataFrame named `structured_data` containing sales data with columns:
["product", "region", "revenue", "online_pct", "profit_margin", "growth_qoq", "avg_order_value"]

Generate a Python one-liner that returns a readable summary (not a raw df dump), such as:
- aggregations
- filters
- sorted views
- value comparisons

Wrap the output in `result = ...` and DO NOT print anything. Assume `structured_data` is already defined.
"""

    code = llm.invoke(prompt).content.strip()
    code = strip_code_fence(code)
    logger.info(f"[Structured Query Code] {code}")

    try:
        # Safe local context for eval
        local_vars = {"structured_data": structured_data.copy()}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "No result.")
        return {
            "answer": str(result),
            "tool": "structured_data_lookup",
            "tool_trace": [source_tool, "structured_data_lookup"],
            "generated_code": code
        }
    except Exception as e:
        logger.error(f"Structured data execution failed: {e}")
        return {
            "answer": "Structured query failed. Could not interpret your question properly.",
            "tool": "structured_data_lookup",
            "tool_trace": [source_tool, "structured_data_lookup"]
        }


def strip_code_fence(code_block: str) -> str:
    """
    Removes ```python ... ``` or ``` ... ``` from a code block string.

    Args:
        code_block (str): The input string containing code block fencing.

    Returns:
        str: Cleaned Python code string without backticks or language markers.
    """
    lines = code_block.strip().splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

tool_doc_search = Tool(
    name="document_search",
    func=document_qa_tool,
    description=(
        "Use this tool if the query involves retrieving specific facts or values directly from the document, "
        "e.g., 'What is the revenue in Q4?', 'Who approved the contract?'. Avoid summarization-style queries."
    )
)

tool_summarizer = Tool(
    name="summarization_tool",
    func=summarize_doc,
    description=(
        "Use this tool when the user's question involves summarizing, extracting key points, generating insights, or "
        "providing a high-level overview of the uploaded documents. "
        "Typical phrases include: 'summarize', 'overview', 'key points', 'insights', 'takeaways', 'high-level summary', "
        "'recommendations', or 'what is the summary of ...'. "
        "Avoid using this for direct factual lookups — use `document_search` in that case."
    )
)

tool_structured = Tool(
    name="structured_data_lookup",
    func=query_structured_data,
    description=(
        "Use this tool when user query goes beyond document and requires access to tabular/structured data."
        "To reiterative, invoke this tool, when the data retrieved from the documents doesn't answer the "
        "question asked"
        "Mention source as the internal structured DataFrame used in memory."
    )
)


tools = [tool_doc_search, tool_summarizer]