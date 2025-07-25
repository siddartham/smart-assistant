import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .logger import logger

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# def index_document(file_path):
#     loader = PyMuPDFLoader(file_path)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)
#     db = FAISS.from_documents(chunks, embedding=EMBEDDINGS)
#     db.save_local(DB_DIR)

def load_and_index(file_paths):
    logger.info(f"Indexing files: {file_paths}")
    docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local("vectorstore/index")
    logger.info(f"Indexed {len(chunks)} chunks.")
    return f"Indexed {len(chunks)} document chunks."
