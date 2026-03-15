from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import Chroma
from chromadb.config import Settings
from chromadb import Client
from langchain_ollama import OllamaEmbeddings

pdfs_directory = "pdfs/"

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunk = text_splitter.split_documents(documents)
    return chunk

def create_embeddings(chunks):
    embeddings = OllamaEmbeddings(model="deepseek-r1")
    return embeddings.embed_query(chunks.page_content)
    
def create_vectors(chunks, embeddings):
    client = Client(Settings())
    collection = client.create_collection(name="dataframe collection")
    # here we are adding docs and embedding to chromadb
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"id": idx}],
            embeddings=[embeddings[idx]],
            ids=[str(idx)] # here we are ensuring ids are string
        )
    retriever = collection.as_retriever()
    return retriever

def query_vectors(input_docs, policy_docs):
    
    pass




