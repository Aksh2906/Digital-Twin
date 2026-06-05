from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from rank_bm25 import BM25Okapi
import pickle
import shutil
import os

load_dotenv()

SOURCE_TITLES = {
    "surely_youre_joking.txt": "Surely You're Joking, Mr. Feynman!",
    "feynman_lectures_vol1.txt": "The Feynman Lectures on Physics, Vol. 1",
    "feynman_nobel_lecture.txt": "Nobel Lecture: The Development of the Space-Time View of QED",
    "feynman_interviews.txt": "Feynman Interviews"
}

loader = DirectoryLoader("data/raw/", glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents")

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95.0
)
chunks = semantic_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    source_file = os.path.basename(chunk.metadata.get("source", "unknown"))
    chunk.metadata["source_title"] = SOURCE_TITLES.get(source_file, source_file)
    chunk.metadata["chunk_index"] = i
    chunk.metadata["preview"] = chunk.page_content[:100]

if os.path.exists("chroma_db_2"):
    shutil.rmtree("chroma_db_2")

db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db_2")
print("ChromaDB created")

corpus = [chunk.page_content for chunk in chunks]
tokenized = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

bm25_data = {
    "bm25": bm25,
    "documents": [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
}

with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25_data, f)

print(f"Done! {len(chunks)} chunks indexed in both ChromaDB and BM25.")