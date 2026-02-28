import streamlit as st

uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
if uploaded_file is not None:
    st.write("File uploaded successfully!")

