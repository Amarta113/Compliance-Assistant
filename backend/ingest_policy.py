from langchain_community.document_loaders import PDFPlumberLoader
from chain import load_and_chunk, create_vectors
from langchain_ollama import OllamaEmbeddings
import os

POLICY_DIR = "pdf/"
DB_DIR = "./backend/db"

def ingest_policies():
    all_chunks = []
    for file in os.listdir(POLICY_DIR):
        if file.endswith(".pdf"):
            chunks = load_and_chunk(os.path.join(POLICY_DIR, file))
            all_chunks.extend(chunks)
    
    vectors = create_vectors(all_chunks)
    print(f"Stored {len(all_chunks)} policy chunks")

if __name__ == "__main__":
    ingest_policies()