from langchain_community.document_loaders import PDFPlumberLoader
from chain import create_chain, setup_retriever
import json

def rag_pipeline(input_pdf_path):
    loader = PDFPlumberLoader(input_pdf_path)
    pages = loader.load()
    full_text = "\n\n".join([p.page_content for p in pages])

    retriever = setup_retriever()
    compliance_chain = create_chain(retriever)
    
    result = compliance_chain.invoke(full_text)
    return result

if __name__ == "__main__":
    input_file = r"C:\Users\hp\Downloads\input_sample_text.pdf"
    report = rag_pipeline(input_file)
    print(json.dumps(report, indent=2))

