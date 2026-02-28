**The Problem:** Researchers want to share genomic data responsibly, but mapping their
specific consent forms against the dense legal text of GA4GH frameworks (like the
Framework for Responsible Sharing) is manual, slow, and error-prone.

**The Solution:** We want to build a lightweight “Compliance Assistant” that does the heavy
lifting. The goal is to create a RAG (Retrieval-Augmented Generation) tool where a
researcher can upload their project’s data use letter, and the AI will check it against
GA4GH standards to flag gaps.

**The Work:** The student will build a Python-based pipeline using open-source LLMs (e.g.,
Llama 3 or Mistral). The core challenge is not just chatting with the document, but
ensuring the bot cites the exact clause in the GA4GH policy that supports its advice. This
The tool will lower the barrier for researchers globally to adopt our standards.
