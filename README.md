# Compliance-Assistant

A lightweight AI-powered tool that helps researchers ensure their genomic data sharing practices comply with GA4GH (Global Alliance for Genomics and Health) standards and frameworks.

## Problem Statement

Researchers want to share genomic data responsibly, but mapping their specific consent forms against the dense legal text of GA4GH frameworks (like the Framework for Responsible Sharing) is:
- **Manual**: Requires careful reading of complex legal documents
- **Slow**: Time-consuming compliance review process
- **Error-prone**: High risk of missing important compliance gaps

## Solution

The Compliance Assistant is a **Retrieval-Augmented Generation (RAG)** tool that automates compliance checking. Researchers can upload their project's data use letter, and the AI will:
- ✅ Check it against GA4GH standards
- ✅ Flag compliance gaps
- ✅ Cite exact clauses in GA4GH policies that support recommendations

## Key Features

- **Document Upload**: Simple PDF upload interface via Streamlit
- **RAG Pipeline**: Leverages open-source LLMs (Llama 3,) for intelligent analysis
- **Citation Support**: AI provides references to specific GA4GH policy clauses
- **Vector Database**: Persistent ChromaDb vector store for efficient retrieval
- **API Backend**: FastAPI integration for scalable deployments
- **JSON Output**: Structured compliance reports for programmatic use

