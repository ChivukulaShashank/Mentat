import os
import faiss
import numpy as np
from gpt4all import Embed4All
from dotenv import load_dotenv

load_dotenv()

class MentatEmbedder:
    def __init__(self):
        # Initialize the embedder
        self.embedder = Embed4All()
        self.kb_file = os.getenv("KB_FILE", "knowledge_base.txt")
        self.index_file = os.getenv("INDEX_FILE", "kb.index")

    def create_and_save_index(self):
        """Reads knowledge base, embeds, and saves FAISS index."""
        if not os.path.exists(self.kb_file):
            raise FileNotFoundError(f"Knowledge base not found: {self.kb_file}")

        with open(self.kb_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        print(f"Embedding {len(sentences)} sentences...")
        
        # Batch embedding is generally more efficient; 
        # for now, this list comprehension preserves your logic
        embeddings = np.array([self.embedder.embed(s) for s in sentences], dtype="float32")
        
        # Build Index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, self.index_file)
        print(f"Successfully saved FAISS index to {self.index_file}")