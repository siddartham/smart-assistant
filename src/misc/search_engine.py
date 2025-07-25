import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
DB_DIR = "../faiss_index"
STRUCTURED_CSV = "worldbank_data.csv"
EMBEDDINGS = OpenAIEmbeddings()
LLM = ChatOpenAI(temperature=0)

# --- Document Indexing ---
def index_document(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embedding=EMBEDDINGS)
    db.save_local(DB_DIR)

# --- Tools ---
@tool
def search_doc(query: str) -> str:
    """Search the uploaded document for relevant content"""
    try:
        db = FAISS.load_local(DB_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "NO_RELEVANT_INFO"
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"[Search Error] {str(e)}"

# Define summarize_doc as single-input compatible tool
summarize_doc_tool = Tool(
    name="summarize_doc",
    func=lambda query: summarize_doc_logic(),
    description="Summarizes the uploaded document."
)

async def summarize_doc_logic():
    try:
        db = FAISS.load_local(DB_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)
        docs = db.similarity_search("summary", k=5)
        content = "\n".join([doc.page_content for doc in docs])
        return LLM.ainvoke(f"Summarize the following:\n\n{content}")
    except Exception as e:
        return f"[Summary Error] {str(e)}"

@tool
def query_structured_source(query: str) -> str:
    """Query a structured dataset as fallback"""
    try:
        df = pd.read_csv(STRUCTURED_CSV)
        return f"Showing first 3 rows of structured data:\n\n{df.head(3).to_markdown(index=False)}"
    except Exception as e:
        return f"[Structured Source Error] {str(e)}"

# --- Streamlit App ---
st.set_page_config(page_title="Smart Assistant", layout="wide")
st.title("🧠 Smart Assistant with Contextual Memory")

# Upload Document
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
if uploaded_file:
    with open("../alt_3/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    index_document("../alt_3/temp.pdf")
    st.success("✅ Document indexed!")

# Ensure structured data exists
if not os.path.exists(STRUCTURED_CSV):
    df = pd.DataFrame({
        "Region": ["North", "South", "East", "West"],
        "Online_Sales_%": [30, 25, 27, 29],
        "Avg_Order_Value": [75.3, 68.9, 71.4, 70.1],
        "Customer_Satisfaction_Score": [8.9, 8.4, 8.6, 8.5],
        "Top_Product": ["Product A", "Product B", "Product C", "Product C"],
        "Units_Sold_Product_A": [2200, 1850, 1700, 1600],
        "Units_Sold_Product_B": [2000, 1900, 1750, 1800],
        "Units_Sold_Product_C": [2100, 1700, 1950, 2000],
        "Revenue_Product_A": [169855, 145000, 132000, 128000],
        "Revenue_Product_B": [155304, 144000, 138000, 139000],
        "Revenue_Product_C": [160141, 135000, 150000, 155000],
        "Return_Rate_%": [2.5, 3.1, 2.9, 3.0],
        "Promo_Code_Usage_%": [12, 15, 13, 14],
        "Peak_Sale_Month": ["December", "November", "November", "December"]
    })
    df.to_csv(STRUCTURED_CSV, index=False)

# Session Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )

if "agent" not in st.session_state:
    tools = [search_doc, summarize_doc_tool, query_structured_source]
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=LLM,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=st.session_state.memory,
        verbose=True
    )

# Query UI
query = st.text_input("Ask your question:")
if query:
    with st.spinner("Thinking..."):
        response = st.session_state.agent.run(query)
        st.markdown("### 🤖 Response")
        st.write(response)

# Show chat history (optional)
with st.expander("🧠 Show chat memory"):
    for msg in st.session_state.memory.chat_memory.messages:
        role = "👤" if msg.type == "human" else "🤖"
        st.markdown(f"**{role}**: {msg.content}")
