import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ner import MentatNER
from src.embed import MentatEmbedder
from src.chat import run_chat_loop

def main():
    print("--- Mentat Pipeline Initialized ---")
    
    # 1. NER Extraction (Matches 'process' method in src/ner.py)
    print("\n[1/3] Extracting Entities...")
    ner = MentatNER()
    ner.process("data/example.pdf", output_txt="knowledge_base.txt")
    
    # 2. Embedding/Indexing
    print("\n[2/3] Building Search Index...")
    embedder = MentatEmbedder()
    embedder.create_and_save_index()
    
    # 3. Chat
    print("\n[3/3] Launching Chatbot...")
    run_chat_loop()

if __name__ == "__main__":
    main()