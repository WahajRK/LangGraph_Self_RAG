📝 Repository Description for GitHub
Self-RAG Streamlit App - AI-Powered Q&A using LangGraph & Pinecone 🚀

📌 Overview
This is a Self-Retrieval Augmented Generation (Self-RAG) system built with:

LangGraph for structured retrieval workflow
Pinecone as the vector database
Groq (Mixtral-8x7b-32768) as the LLM
Google Generative AI for embeddings
Streamlit for an interactive UI
This application allows users to ask questions, retrieves relevant documents from Pinecone, and generates accurate AI-powered responses while checking for hallucinations.

✨ Features
✅ Self-RAG Workflow: Automated query refinement, retrieval, grading, and response generation.
✅ Hallucination Detection: Ensures AI-generated responses are fact-based.
✅ Query Rewriting: Enhances retrieval for improved answers.
✅ Interactive UI: Built with Streamlit for real-time AI Q&A.
✅ Pinecone Vector Search: Uses MMR (Maximal Marginal Relevance) for intelligent document retrieval.
✅ Fast & Scalable: Optimized with LangGraph stateful processing.

🛠️ Tech Stack
🔹 Python (Streamlit)
🔹 LangGraph (StateGraph-based RAG pipeline)
🔹 Pinecone (Vector database for semantic search)
🔹 Google Generative AI (Embeddings for document retrieval)
🔹 Groq LLM (Mixtral-8x7b) (LLM for response generation)

![image](https://github.com/user-attachments/assets/15df3123-cbea-4f3f-9968-f795000cde38)

STEPS TO RUN:
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
python parser.py (To add pdf to the vector store)
python langgrap_rag.py (For termial base application)
streamlit run app.py (for browser base application)

![image](https://github.com/user-attachments/assets/7d1e9f09-516b-42ab-a7f2-dc57ce5932b5)



