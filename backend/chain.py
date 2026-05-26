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

def load_input_document(input_pdf_path):
    """Load and chunk the INPUT research document for compliance review"""
    logger.info(f"Loading input research document: {input_pdf_path}")
    if not os.path.exists(input_pdf_path):
        logger.error(f"Input file not found: {input_pdf_path}")
        return None

    input_chunks = load_and_chunk(input_pdf_path)
    logger.info(f"Loaded and chunked {len(input_chunks)} input document chunks")
    return input_chunks

def create_chain(retriever):
    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        temperature=0,
        groq_api_key="add-your-groq-api-key-here"
    )

    compliance_chain = (
        RunnableParallel({
            "research_text": RunnablePassthrough(),
            "context": (
                retriever
            )
        }) 
        | compliance_prompt
        | llm
        | JsonOutputParser()
    )
    logger.info("Compliance chain created successfully.")
    return compliance_chain

