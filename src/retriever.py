import os
import faiss
import numpy as np
from gpt4all import Embed4All
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

class MentatRetriever:
    def __init__(self):
        print("Loading retriever resources...")
        self.embedder = Embed4All()
        self.index_path = os.getenv("INDEX_FILE", "kb.index")
        self.kb_path = os.getenv("KB_FILE", "knowledge_base.txt")
        
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file {self.index_path} not found. Run embed.py first.")
            
        self.index = faiss.read_index(self.index_path)
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.kb_chunks = [line.strip() for line in f if line.strip()]

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        arr = np.array([self.embedder.embed(t) for t in texts], dtype="float32")
        faiss.normalize_L2(arr)
        return arr

    def broad_search_per_candidate(self, candidates: List[str], broad_k: int = 50) -> List[int]:
        cand_embs = self.embed_texts(candidates)
        union_set = set()
        for i in range(cand_embs.shape[0]):
            q = cand_embs[i:i+1]
            D, I = self.index.search(q, broad_k)
            union_set.update([int(x) for x in I[0] if x != -1])
        return list(union_set)

    def refined_search_question(self, question: str, candidate_indices: List[int], refined_k: int = 8) -> List[Tuple[float, int, str]]:
        if not candidate_indices:
            return []
        
        q_emb = self.embed_texts([question])
        candidate_texts = [self.kb_chunks[i] for i in candidate_indices]
        cand_embs = self.embed_texts(candidate_texts)
        
        sims = (cand_embs @ q_emb.T).squeeze()
        top_k = min(refined_k, len(candidate_indices))
        top_idx = np.argsort(-sims)[:top_k]
        
        return [(float(sims[ti]), candidate_indices[ti], self.kb_chunks[candidate_indices[ti]]) for ti in top_idx]

    def topk_global(self, question: str, k: int = 8) -> List[Tuple[float, int, str]]:
        q_emb = self.embed_texts([question])
        D, I = self.index.search(q_emb, k)
        return [(float(sc), int(idx), self.kb_chunks[idx]) for sc, idx in zip(D[0], I[0]) if idx != -1]