import os
import time
from dotenv import load_dotenv
from uuid import uuid4
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def parse_pdf(pdf_directory="./data"):
    """ Parses PDFs and extracts text using LlamaParse. """
    parser = LlamaParse(result_type="markdown")  
    file_extractor = {".pdf": parser}
    
    documents = SimpleDirectoryReader(input_dir=pdf_directory, file_extractor=file_extractor).load_data()
    
    if not documents:
        raise ValueError("❌ No documents loaded. Ensure the directory contains PDFs.")

    all_text = "\n\n".join([doc.text_resource.text for doc in documents if doc.text_resource])
    
    if not all_text.strip():
        raise ValueError("❌ No text extracted from documents. Check LlamaParse configuration.")

    return all_text

def chunk_text(text, chunk_size=1024, chunk_overlap=128):
    """ Splits text into manageable chunks. """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", ".", " "]
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        raise ValueError("❌ No chunks created. Check text splitting configuration.")
    
    return chunks

def generate_embeddings(chunks):
    """ Generates embeddings using GoogleGenerativeAI. """
    google_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
    embedded_chunks = google_embeddings.embed_documents(chunks)

    if not embedded_chunks or len(embedded_chunks) != len(chunks):
        raise ValueError("❌ Embedding generation failed. Check Google API key or model configuration.")

    return google_embeddings, embedded_chunks

def setup_pinecone(index_name="byte-corp", dimension=768):
    """ Initializes Pinecone and creates an index if needed. """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("❌ Pinecone API key not found.")

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"⚡ Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc, pc.Index(index_name)

def store_embeddings_in_pinecone(chunks, google_embeddings, index):
    """ Stores chunk embeddings in Pinecone. """
    vector_store = PineconeVectorStore(index=index, embedding=google_embeddings)
    documents = [Document(page_content=chunk, metadata={"chunk_id": i, "source": "PDF Document"}) for i, chunk in enumerate(chunks)]
    uuids = [str(uuid4()) for _ in range(len(documents))]
    
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store

def main():
    """ Main function to parse PDFs, chunk text, generate embeddings, and store them in Pinecone. """
    try:
        print("🚀 Parsing PDFs...")
        text = parse_pdf()
        
        print("📜 Splitting text into chunks...")
        chunks = chunk_text(text)
        
        print("🔢 Generating embeddings...")
        google_embeddings, _ = generate_embeddings(chunks)
        
        print("🗄️ Setting up Pinecone...")
        _, index = setup_pinecone()
        
        print("📥 Storing embeddings in Pinecone...")
        store_embeddings_in_pinecone(chunks, google_embeddings, index)
        
        print("✅ Pinecone vector database setup completed!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
