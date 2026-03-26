from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

pdfs_directory = "pdf/"

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_and_chunk(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    for i, doc in enumerate(documents):
        doc.metadata["source"] = file_path
        doc.metadata["page"] = i + 1

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunk = text_splitter.split_documents(documents)
    return chunk

def create_vectors(chunks):
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectors = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./backend/db")
    print("Policy vectors stored successfully")

def load_retriever():
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    _vectors = Chroma(
        embedding_function=embeddings_model,
        persist_directory="./backend/db"
    )
    return _vectors.as_retriever(search_kwargs={"k": 5})
