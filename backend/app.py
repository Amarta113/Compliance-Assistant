import streamlit as st
from pipeline import rag_pipeline
import requests


API_BASE_URL = "http://localhost:8000"

st.title("Agentic Compliance AI")
uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

if st.button("Comply"):
    if uploaded_file:
        with st.spinner("Running Compliance Agents..."):
            response = requests.post(f"{API_BASE_URL}/analyze", files={"file": uploaded_file})
            result = response.json().get("report", "No report generated.")            
            if result: 
                st.subheader("Final Report")
                st.write(result)

