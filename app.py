import streamlit as st
import logging
from langgraph_rag import run_rag_query

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit UI Setup
st.set_page_config(page_title="Self-RAG App", layout="wide")
st.title("🔍 Self-RAG: AI-Powered Q&A")
st.write("Enter a query and get AI-generated answers based on retrieved documents.")

# Input Section
question = st.text_input("Ask your question:", placeholder="Type your question here...")
submit_button = st.button("Get Answer")

# Output Display
if submit_button and question:
    with st.spinner("Processing your query... ⏳"):
        response = run_rag_query(question)
    
    # Display Response
    st.subheader("🤖 AI Response:")
    st.write(response)

    # Logging
    logging.info(f"User Query: {question}")
    logging.info(f"AI Response: {response}")
