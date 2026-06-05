import pickle
import os
import hashlib
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

BM25_INDEX_PATH = "bm25_index.pkl"


class HybridRetriever:
    def __init__(self, chroma_dir="chroma_db_2", bm25_path=BM25_INDEX_PATH,
                 embedding_model="BAAI/bge-small-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db = Chroma(persist_directory=chroma_dir, embedding_function=self.embeddings)

        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.bm25_docs = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in bm25_data["documents"]
            ]
        else:
            self.bm25 = None
            self.bm25_docs = []

    def retrieve(self, queries, semantic_k=20, bm25_k=20, final_k=10):
        if isinstance(queries, str):
            queries = [queries]

        all_semantic = []
        all_bm25 = []

        for query in queries:
            sem_docs = self.db.similarity_search(query, k=semantic_k)
            all_semantic.extend(sem_docs)

            if self.bm25 is not None:
                tokenized = query.lower().split()
                scores = self.bm25.get_scores(tokenized)
                top_indices = scores.argsort()[-bm25_k:][::-1]
                bm25_hits = [self.bm25_docs[i] for i in top_indices if scores[i] > 0]
                all_bm25.extend(bm25_hits)

        all_semantic = self._deduplicate(all_semantic)
        all_bm25 = self._deduplicate(all_bm25)

        if self.bm25 is not None and all_bm25:
            fused = self._rrf_fusion(all_semantic, all_bm25)
        else:
            fused = all_semantic

        return fused[:final_k]

    @staticmethod
    def _doc_id(doc):
        return hashlib.md5(doc.page_content.encode()).hexdigest()

    def _deduplicate(self, docs):
        seen = set()
        unique = []
        for doc in docs:
            did = self._doc_id(doc)
            if did not in seen:
                seen.add(did)
                unique.append(doc)
        return unique

    def _rrf_fusion(self, semantic_docs, bm25_docs, k=60):
        doc_scores = {}

        for rank, doc in enumerate(semantic_docs):
            did = self._doc_id(doc)
            if did not in doc_scores:
                doc_scores[did] = {"score": 0, "doc": doc}
            doc_scores[did]["score"] += 1 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            did = self._doc_id(doc)
            if did not in doc_scores:
                doc_scores[did] = {"score": 0, "doc": doc}
            doc_scores[did]["score"] += 1 / (k + rank + 1)

        ranked = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked]
