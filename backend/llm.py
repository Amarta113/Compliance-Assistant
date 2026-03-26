from chain import create_chain, load_and_chunk, setup_retriever

def rag_pipeline(docs):
    chunks = load_and_chunk(docs)
    retriever = setup_retriever()
    compliance_chain = create_chain(retriever)
    results = []
    for chunk in chunks:
        result = compliance_chain.invoke(chunk.page_content)
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

if __name__ == "__main__":
    input_file = r"C:\Users\hp\Downloads\s40142-019-00164-9.pdf"  # Research document to review
    report = rag_pipeline(input_file)
    print(report)
