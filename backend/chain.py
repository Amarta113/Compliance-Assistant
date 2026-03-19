from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from prompt import compliance_prompt; 
from vector_db import create_vectors, create_embeddings, load_and_chunk
from langchain_ollama import ollama 
from langchain_groq import ChatGroq



def format_docs(docs):
    format = []
    for i , doc in enumerate(docs, 1):
        location = docs.metadatas.get("source": "")
        format.append(
            f"[{i}] {location} \n {doc.page_content}"
        )
    return "\n\n".join(fromat)

def chain(docs):
    format_context = RunnableLambda(format_docs)
    chunks = load_and_chunk(docs)
    embeddings = create_embeddings(docs)
    retriever = create_vectors(chunks, embeddings)
    #llm = ChatOllama(
    #    model="deepseek-r1",
    #    temperature=0
    #    format=JSON
    #    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        format=JSON,
        groq_api_key=""
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

    return chain
