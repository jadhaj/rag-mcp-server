# MCP RAG Tool

This project implements a **Retrieval-Augmented Generation (RAG)** system integrated with an **MCP (Model Context Protocol)** server using **FastMCP**.  
It provides tools for semantic search and context retrieval from codebases, specifically targeting **OpenShift test code** and the **Prow CI ecosystem**.


## Features
- **RAG with FAISS Vector Store**: Efficient similarity search over indexed documents.  
- **OpenAI and HuggingFace Embeddings**: Supports multiple embedding models.  
- **FastMCP Server**: Exposes retrieval tools via MCP for integration with other systems.  
- **Document Conversion & Indexing**: Converts source files to text, splits documents, and updates metadata for search.  
- **Specialized Retrieval Tools**:  
  - `retrieve_openshift_tests_context`: Finds relevant Golang/Ginkgo test code for OpenShift.  
  - `retrieve_prow_ci_context`: Finds relevant context for Prow CI ecosystem.  


## Setup

### 1. Install dependencies

pip install -r requirements.txt

### 2. Set environment variables

Required:

OPENAI_API_KEY

Optional:

GROQ_API_KEY (for Groq models)

You can use a .env file or set them in your shell.

### 3. Prepare source data

Update src_dir and dst_dir paths in mcp_rag_tool.py to point to your source code and destination directories.

On startup, the script will:

- Convert files
- documents
- Build FAISS indices


## Usage
Run the MCP RAG server:
python mcp_rag_tool.py
