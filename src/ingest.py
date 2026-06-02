from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import os

load_dotenv()

loader = DirectoryLoader("data/raw/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
semantic_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile",breakpoint_threshold_amount=95.0)
chunks = semantic_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")




db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db_2")
print("Done! Knowledge base created.")