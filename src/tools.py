import pandas as pd
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .indexer import load_and_index
from .logger import logger
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Dummy structured data
structured_data = pd.DataFrame({
    "product": ["Loan A", "Loan B", "Credit Card X"],
    "interest_rate": [7.5, 6.9, 19.9],
    "description": ["Home loan", "Car loan", "Rewards card"]
})


def document_qa_tool(query: str) -> str:
    logger.info(f"[Tool: Document Search] Query: {query}")
    vectordb = FAISS.load_local("vectorstore/index", embedding_model, allow_dangerous_deserialization=True)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    answer = qa_chain.run(query)
    logger.info(f"[Document Search] Answer: {answer}")
    return answer


def query_structured_data(query: str) -> str:
    logger.info(f"[Tool: Structured Data] Query: {query}")
    if "interest rate" in query.lower():
        return structured_data.to_string(index=False)
    elif "loan" in query.lower():
        return structured_data[structured_data["product"].str.contains("loan", case=False)].to_string(index=False)
    return "No relevant structured data found."


def summarize_doc(query: str) -> str:
    logger.info(f"[Tool: Summarization] Query: {query}")
    return document_qa_tool("Summarize: " + query)


tool_doc_search = Tool(
    name="document_search",
    func=document_qa_tool,
    description=(
        "Use this tool for any questions that require details or facts from uploaded documents. "
        "Examples: 'What are the North region sales?', 'Summarize Q4 report', 'Tell me top-selling product in uploaded document'."
    )

)

tool_structured = Tool(
    name="structured_data_lookup",
    func=query_structured_data,
    description="Useful when user query goes beyond document and requires access to tabular/structured data."
)

tool_summarizer = Tool(
    name="summarization_tool",
    func=summarize_doc,
    description="Use this to summarize specific sections or extract key insights from uploaded documents."
)

tools = [tool_doc_search, tool_structured, tool_summarizer]