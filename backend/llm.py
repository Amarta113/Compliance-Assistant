from langchain_ollama import ollama 
from chain import chain, load_and_chunk

def rag_pipeline(docs):
    chunk_size = load_and_chunk(docs)
    results = []
    for chunk in chunks:
        result = chain.invoke(chunk.page_content)
        results.append(result)

    violations = []

    for r in results:
        if not r["compliant"]:
            violations.append(r)

    report = {
        "total_chunks": len(results),
        "violations": violations
    }
    return report