from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from prompt import compliance_prompt
from vector_db import create_vectors, load_and_chunk, load_retriever
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_docs(docs):
    """Format retrieved documents for display"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        location = doc.metadata.get("source","")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[{i}] {location} (Page {page})\n{doc.page_content}"
        )
    logger.info(f"Formatted {len(docs)} documents for context.")
    return "\n\n".join(formatted)


def load_input_document(input_pdf_path):
    """Load and chunk the INPUT research document for compliance review"""
    logger.info(f"Loading input research document: {input_pdf_path}")
    if not os.path.exists(input_pdf_path):
        logger.error(f"Input file not found: {input_pdf_path}")
        return None

    input_chunks = load_and_chunk(input_pdf_path)
    logger.info(f"Loaded and chunked {len(input_chunks)} input document chunks")
    return input_chunks

def setup_retriever():
    retriever = load_retriever()
    logger.info("Retriever setup complete - loaded persistent policy database")
    return retriever

def create_chain(retriever):
    
    format_context = RunnableLambda(format_docs)
    
    """llm = ChatOllama(
        model="deepseek-r1:8b",
        temperature=0,
        response_format={"type": "json_object"},
    )
    """
    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature=0,
        response_format={"type": "json_object"},
        groq_api_key="your_groq_api_key_here"
    )

    compliance_chain = (
        RunnableParallel({
            "research_text": RunnablePassthrough(),
            "context": (
                retriever
                | format_context
            )
        }) 
        | compliance_prompt
        | llm
        | JsonOutputParser()
    )
    logger.info("Compliance chain created successfully.")
    return compliance_chain

