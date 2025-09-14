import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from mcp.server.fastmcp import FastMCP
from utils import convert_files_to_txt, get_embeddings_model, split_documents, update_metadata
from typing import Dict, Any, List

from mcp.types import PromptMessage, TextContent


# ... other imports ...

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("RAGToolServer")

# Configuration (can be moved to a config file for production)
src_dir = "/Users/jhajyahy/Repositories/openshift-tests-private/test/extended/installer/baremetal/"
dst_dir = "/Users/jhajyahy/data/"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- RAG Core Logic Setup ---
# This part of the code initializes the vector store and the RAG chain.
# It is run once when the server starts.

# Load embeddings model
embeddings = get_embeddings_model("OpenAI")

if os.path.exists(src_dir):
    # Convert Files to Text
    convert_files_to_txt(src_dir, dst_dir)
    documents = split_documents(dst_dir)
    update_metadata(documents)

    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local("faiss_rag_index")
    print("Files converted and indexed successfully.")
    
else:
    print(f"Source directory {src_dir} does not exist. Please check the path.")


# Load the FAISS vector database
try:
    db = FAISS.load_local("faiss_rag_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("RAG System initialized with FAISS index.")
except RuntimeError as e:
    print(f"Error loading FAISS index: {e}")
    print("Please run the file conversion and index creation script first.")
    exit(1)

src_dir = "/Users/jhajyahy/Repositories/release/ci-operator/step-registry/baremetal/lab/pre/"
dst_dir = "/Users/jhajyahy/prow-data/"

if os.path.exists(src_dir):
    # Convert Files to Text
    convert_files_to_txt(src_dir, dst_dir)
    documents = split_documents(dst_dir)
    update_metadata(documents)

    vectordb = FAISS.from_documents(documents, embeddings)
    vectordb.save_local("faiss_prow_index")
    print("Files converted and indexed successfully.")
    
else:
    print(f"Source directory {src_dir} does not exist. Please check the path.")

# Load the FAISS vector database
try:
    db = FAISS.load_local("faiss_prow_index", embeddings, allow_dangerous_deserialization=True)
    prow_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("RAG System initialized with FAISS index.")
except RuntimeError as e:
    print(f"Error loading FAISS index: {e}")
    print("Please run the file conversion and index creation script first.")
    exit(1)

@mcp.tool()
def retrieve_openshift_tests_context(question: str) -> List[Dict[str, Any]]:
    """
    Retrieves and returns relevant document context from the vector database based on a user's question.
    This tool is specialized in finding relevant Golang and Ginkgo framework test code snippets for OpenShift.

    Args:
        question: The user's question to be used for a similarity search.

    Returns:
        A list of dictionaries, where each dictionary contains the content and metadata
        of a retrieved document.
    """
    print(f"Retrieving documents for question: '{question}'...")
    
    # Use the retriever to find relevant documents
    try:
        docs = retriever.get_relevant_documents(question)
        
        # Format the documents into a list of dictionaries to be returned
        retrieved_context = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        
        print(f"Retrieved {len(retrieved_context)} documents.")
        return retrieved_context
        
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return [{"error": str(e)}]

@mcp.tool()
def retrieve_prow_ci_context(question: str) -> List[Dict[str, Any]]:
    """
    Retrieves and returns relevant document context from the vector database based on a user's question.
    This tool is specialized in prow CI ecosystem.

    Args:
        question: The user's question to be used for a similarity search.

    Returns:
        A list of dictionaries, where each dictionary contains the content and metadata
        of a retrieved document.
    """
    print(f"Retrieving documents for question: '{question}'...")
    
    # Use the retriever to find relevant documents
    try:
        docs = prow_retriever.get_relevant_documents(question)
        
        # Format the documents into a list of dictionaries to be returned
        retrieved_context = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        
        print(f"Retrieved {len(retrieved_context)} documents.")
        return retrieved_context
        
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return [{"error": str(e)}]

# Main entry point to run the server
if __name__ == "__main__":
    print("Starting FastMCP server...")
    mcp.run()
