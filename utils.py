import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP


import getpass

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
GROQ_API_KEY = os.environ['GROQ_API_KEY']

def convert_files_to_txt(src_dir, dst_dir):
    # If the destination directory does not exist, create it.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if not file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                # Create the same directory structure in the new directory
                new_root = os.path.join(dst_dir, os.path.dirname(rel_path))
                os.makedirs(new_root, exist_ok=True)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            data = f.read()
                    except UnicodeDecodeError:
                        print(f"Failed to decode the file: {file_path}")
                        continue
                # Create a new file path with .txt extension
                new_file_path = os.path.join(new_root, file + '.txt')
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(data)

# Load and split documents


def split_documents(dst_dir):
    loader = DirectoryLoader(
        dst_dir, show_progress=True, loader_cls=TextLoader)
    repo_files = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150)
    documents = text_splitter.split_documents(documents=repo_files)
    return documents


def update_metadata(documents):
    for doc in documents:
        old_path_with_txt_extension = doc.metadata["source"]
        new_path_without_txt_extension = old_path_with_txt_extension.replace(
            ".txt", "")
        doc.metadata.update({"source": new_path_without_txt_extension})

# Function to get LLM based on selection


def get_llm(model_name):
    llm_map = {
        "GPT-3.5": ChatOpenAI(temperature=0.7),
        "GPT-4": ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-4o",
            streaming=True,
          ),
        "HF": HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            do_sample=False,
            repetition_penalty=1.03,
        ),
        "Model Server": local_model()
    }
    # Default to GPT-3.5 if not found
    return llm_map.get(model_name, ChatOpenAI(temperature=0.7))

def get_embeddings_model(model):
    llm_map = {
        "OpenAI": OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"),
        "HF": hugg_embed_model()
    }
    # Default to GPT-3.5 if not found
    return llm_map.get(model, OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"))


def hugg_embed_model ():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings":True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    )
    return embeddings

def groq_model():
    from langchain_groq import ChatGroq
    llama3 =ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-70b-8192",
        streaming=True
    )
    return llama3

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def local_model():
    # Initialize the local ChatOpenAI instance
    local_llm = ChatOpenAI(
        base_url="http://localhost:8001/v1/",  # Ensure the base URL points to your server's API
        api_key="EMPTY",
        model="granite-7b-lab-Q4_K_M.gguf",      # Model name recognized by the server
        streaming=True                        # Enable streaming if supported by the server
    )
    return local_llm

