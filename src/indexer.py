from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from .logger import logger

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def dynamic_section_split(doc_text: str):
    """
    Split text into sections dynamically using layout heuristics.
    """
    lines = doc_text.splitlines()
    sections = []
    buffer = []
    current_section = []

    def is_probable_header(line: str) -> bool:
        stripped = line.strip()
        return (
            5 < len(stripped) < 80 and
            stripped.istitle() or stripped.isupper() and
            not stripped.endswith(".") and
            len(stripped.split()) <= 8
        )

    for line in lines:
        if is_probable_header(line):
            if current_section:
                sections.append("\n".join(current_section).strip())
                current_section = []
            current_section.append(line.strip())
        else:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section).strip())

    return sections


def load_and_index(file_paths):
    logger.info(f"Indexing files: {file_paths}")
    docs = []

    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            continue
        docs.extend(loader.load())

    all_chunks = []
    for doc in docs:
        dynamic_sections = dynamic_section_split(doc.page_content)
        for sec in dynamic_sections:
            new_doc = doc.copy()
            new_doc.page_content = sec.strip()
            all_chunks.append(new_doc)

    logger.info(f"Total initial dynamic sections: {len(all_chunks)}")

    # Optional: re-split long sections
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_chunks = splitter.split_documents(all_chunks)

    vectordb = FAISS.from_documents(final_chunks, embedding_model)
    vectordb.save_local("vectorstore/index")

    logger.info(f"Indexed {len(final_chunks)} chunks after splitting.")
    return f"Indexed {len(final_chunks)} layout-aware chunks from uploaded files."
