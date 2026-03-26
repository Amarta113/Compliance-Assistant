from chain import create_chain, load_and_chunk, setup_retriever

def rag_pipeline(docs):
    chunks = load_and_chunk(docs)
    retriever = setup_retriever()
    compliance_chain = create_chain(retriever)
    results = []
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_text = "\n\n".join([c.page_content for c in batch])
        result = compliance_chain.invoke(batch_text)
        results.append(result)

    violations = []

    
    for r in results:
        if isinstance(r, list):  
            for item in r:
                if not item.get("compliant", True):
                    violations.append(item)
        else:  
            if not r.get("compliant", True):
                violations.append(r)

    report = {
        "total_chunks": len(results),
        "violations": violations
    }
    return report

if __name__ == "__main__":
    input_file = r"C:\Users\hp\Downloads\s40142-019-00164-9.pdf"  # Research document to review
    report = rag_pipeline(input_file)
    print(report)
