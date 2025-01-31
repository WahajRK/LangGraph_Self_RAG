# langgraph_rag.py (Self-RAG with Debug Logs & Improvements)
import os
import logging
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define Graph State
class GraphState(TypedDict):
    """Stores question, retrieved documents, and generated response."""
    question: str
    retrieved_docs: List[Document]
    generation: str
    requery: bool
    query_rewrite_count: int

# Step 1: Initialize Pinecone & Vector Store
def setup_pinecone(index_name="bytecorp"):
    """ Connects to Pinecone vector store. """
    logging.info("🔹 Connecting to Pinecone...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        logging.error("❌ Pinecone API key not found.")
        raise ValueError("❌ Pinecone API key not found.")

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        logging.error("❌ Pinecone index not found. Run parser.py first.")
        raise ValueError("❌ Pinecone index not found. Run parser.py first.")

    logging.info(f"✅ Connected to Pinecone index: {index_name}")
    return pc.Index(index_name)

# Step 2: Initialize Models (Google Embeddings & Groq LLM)
def initialize_models():
    """ Initializes GoogleGenerativeAI embeddings and Groq LLM. """
    logging.info("🔹 Initializing Google Generative AI Embeddings & Groq LLM...")
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
    groq_llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, max_tokens=1024, max_retries=2)
    logging.info("✅ Models initialized successfully!")
    return google_embeddings, groq_llm

# Step 3: Document Retrieval using MMR
def retrieve(state: GraphState):
    """Retrieves relevant documents from Pinecone using MMR."""
    logging.info(f"📥 Retrieving documents for query: '{state['question']}'")
    index = setup_pinecone()
    google_embeddings, _ = initialize_models()
    vector_store = PineconeVectorStore(index=index, embedding=google_embeddings)

    retrieved_docs = vector_store.max_marginal_relevance_search(
        state["question"], k=4, fetch_k=10, lambda_mult=0.7
    )

    if not retrieved_docs:
        logging.warning("⚠️ No documents retrieved!")
    else:
        logging.info(f"✅ Retrieved {len(retrieved_docs)} relevant documents.")

    return {"retrieved_docs": retrieved_docs, "question": state["question"]}

# Step 4: Grade Retrieved Documents (Relevance Check)
def grade_documents(state: GraphState):
    """Determines if retrieved documents are relevant to the query."""
    logging.info("📝 Grading retrieved documents for relevance...")
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]

    if not retrieved_docs:
        logging.warning("⚠️ No relevant documents retrieved, may need to re-query.")
        return {"retrieved_docs": [], "question": question, "requery": True}

    logging.info(f"✅ Retained {len(retrieved_docs)} relevant documents.")
    return {"retrieved_docs": retrieved_docs, "question": question, "requery": False}

# Step 5: Generate Answer using Groq LLM
def generate(state: GraphState):
    """Generates response using retrieved documents & Groq LLM."""
    logging.info("🤖 Generating response using retrieved context...")
    _, groq_llm = initialize_models()
    context = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])

    messages = [
        ("system", "You are an AI assistant that answers questions based on provided documents."),
        ("system", f"Context:\n{context}"),
        ("human", state["question"]),
    ]

    response = groq_llm.invoke(messages)
    logging.info(f"✅ Response generated: {response.content[:100]}... (truncated for brevity)")
    return {"retrieved_docs": state["retrieved_docs"], "question": state["question"], "generation": response.content}

# Step 5.5: Check for Hallucinations
def grade_hallucinations(state: GraphState):
    """Determines if the generated response is grounded in the retrieved documents."""
    logging.info("🔎 Checking for hallucinations in generated response...")

    question = state["question"]
    retrieved_docs = "\n\n".join([doc.page_content for doc in state["retrieved_docs"]])  # Convert docs to clean text
    generation = state["generation"]

    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fact-checking AI verifying if an answer is supported by retrieved documents."),
        ("human", "Set of facts: \n\n {documents} \n\n AI-generated response: {generation} \n\n "
                  "Does the response contain only facts from the retrieved documents? Answer 'Yes' or 'No'."),
    ])

    # ✅ Use proper placeholders
    prompt_messages = hallucination_prompt.format_messages(documents=retrieved_docs, generation=generation)

    _, groq_llm = initialize_models()
    response = groq_llm.invoke(prompt_messages)

    if "no" in response.content.lower():
        logging.warning("🚨 Detected potential hallucination! Rerunning retrieval...")
        return {"retrieved_docs": state["retrieved_docs"], "question": state["question"], "requery": True, "next_step": "transform_query"}

    logging.info("✅ No hallucination detected, finalizing response.")
    return {"retrieved_docs": state["retrieved_docs"], "question": state["question"], "generation": state["generation"], "requery": False, "next_step": "finalize_response"}

# Step 5.75: Answer Grading
def grade_answer(state: GraphState):
    """Determines if the generated response sufficiently answers the query."""
    logging.info("🔎 Checking if response fully addresses the query...")

    question = state["question"]
    generation = state["generation"]

    # ✅ Define proper prompt template
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict evaluator assessing if an AI-generated response fully resolves a query."),
        ("human", "Query: {question} \n\n Answer: {generation} \n\n "
                  "Does this answer fully satisfy the query? Answer 'Yes' or 'No'."),    
    ])

    # ✅ Use format_messages() to correctly structure the input
    prompt_messages = answer_prompt.format_messages(question=question, generation=generation)

    _, groq_llm = initialize_models()
    response = groq_llm.invoke(prompt_messages)

    if "no" in response.content.lower():
        logging.warning("🚨 Answer does not fully satisfy the query. Refining response...")
        return {"retrieved_docs": state["retrieved_docs"], "question": state["question"], "requery": True, "next_step": "transform_query"}

    logging.info("✅ Answer sufficiently addresses the query.")
    return {"retrieved_docs": state["retrieved_docs"], "question": state["question"], "generation": state["generation"], "requery": False, "next_step": "finalize_response"}

# Step 6: Transform Query (Rewrites for Better Retrieval)
def transform_query(state: GraphState):
    """Rewrites the query for improved retrieval. Prevents infinite loops."""
    if state["query_rewrite_count"] >= 3:
        logging.warning("🚨 Query transformation limit reached. Returning 'no relevant data' response.")
        return {
            "question": state["question"],  # Keep the original question
            "retrieved_docs": [],  # No docs found
            "generation": "Sorry, I couldn't find relevant information for your query.",
            "requery": False,  # Stop the loop
            "next_step": "finalize_response",
        }

    logging.info(f"🔄 Transforming query (Attempt {state['query_rewrite_count'] + 1})...")
    
    _, groq_llm = initialize_models()
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI that rewrites user queries to improve search results."),
        ("human", f"Rewrite the following question to make it more precise and effective for retrieval:\n\n {state['question']}")
    ])
    
    rewritten_query = groq_llm.invoke(rewrite_prompt.format_messages()).content.strip()
    
    logging.info(f"✅ New rewritten query: {rewritten_query}")

    return {
        "question": rewritten_query,
        "retrieved_docs": [],
        "query_rewrite_count": state["query_rewrite_count"] + 1,  # ✅ Increment count
        "requery": True,
        "next_step": "retrieve",
    }


# Step 7: Decide Whether to Generate or Transform Query
def decide_generation(state: GraphState):
    """Decides whether to generate an answer or rewrite the query."""
    if state["requery"]:
        if state["query_rewrite_count"] >= 3:
            logging.warning("🚨 Query transformation exhausted. Providing best available response.")
            return "finalize_response"
        logging.info("🚨 No relevant documents found, transforming query...")
        return "transform_query"
    
    logging.info("✅ Sufficient documents found, proceeding to generation...")
    return "generate"


# Step 8: Build LangGraph Workflow
workflow = StateGraph(GraphState)

# Define Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("grade_hallucinations", grade_hallucinations)
workflow.add_node("grade_answer", grade_answer)

# Define Workflow Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_generation, {
    "transform_query": "transform_query",
    "generate": "generate",
})
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", "grade_hallucinations")
workflow.add_conditional_edges("grade_hallucinations", lambda state: state["next_step"], {
    "transform_query": "transform_query",
    "finalize_response": END,
})
workflow.add_conditional_edges("grade_answer", lambda state: state["next_step"], {
    "transform_query": "transform_query",
    "finalize_response": END,
})

# Compile the Workflow
app = workflow.compile()

# Step 9: Run the LangGraph Workflow with Logging
def run_rag_query(question: str):
    """Runs the LangGraph workflow with a given question."""
    logging.info(f"\n🔍 Running Self-RAG for: {question}")
    
    # ✅ Ensure query_rewrite_count is initialized
    inputs = {
        "question": question,
        "retrieved_docs": [],  # Initialize as empty
        "generation": "",  # Initialize empty response
        "requery": False,  # Default state
        "query_rewrite_count": 0,  # ✅ Initialize rewrite counter
    }

    for output in app.stream(inputs):
        for key, value in output.items():
            logging.info(f"🔗 Node: {key}")
            if key == "retrieve":
                logging.info(f"📜 Retrieved Docs: {[doc.page_content[:50] + '...' for doc in value['retrieved_docs']]}")  # Truncated
            elif key == "generate":
                logging.info(f"🤖 Generated Response: {value['generation'][:100]}...")  # Truncated

    return value["generation"]


if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Ask a question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            logging.info("👋 Exiting.")
            break
        response = run_rag_query(user_query)
        print(f"\n🤖 **AI Response:** {response}")



