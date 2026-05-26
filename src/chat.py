import os
from dotenv import load_dotenv
from gpt4all import GPT4All
from src.retriever import MentatRetriever

load_dotenv()

# -------- CONFIG --------
# Construct an absolute path relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Moves up one directory from 'src' to project root, then into 'models'
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", "mistral-7b-instruct-v0.2.Q4_0.gguf")

LLM_N_CTX = int(os.getenv("LLM_N_CTX", 1024))
BROAD_K = 40
REFINED_K = 8
TOP_K = 3

def get_llm():
    """Lazy-loads the LLM using an absolute path to ensure file location is found."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    # Passing the absolute path directly forces GPT4All to use this file
    return GPT4All(MODEL_PATH, allow_download=False, n_ctx=LLM_N_CTX)

def run_mkrag_mcq(question, candidates, broad_k=BROAD_K, refined_k=REFINED_K):
    retriever = MentatRetriever()
    llm = get_llm()
    # Your logic here
    pass

def run_open_query(question, top_k=TOP_K):
    retriever = MentatRetriever()
    llm = get_llm()
    # Your logic here
    pass

def run_chat_loop():
    """Main execution loop for the chatbot."""
    print("Mentat RAG Chatbot (MCQ + Open QA). Type 'quit' to exit.")
    
    while True:
        mode = input("\nEnter mode (mcq/open): ").strip().lower()
        if mode in ["quit", "exit"]:
            print("Shutting down Mentat...")
            break
        
        if mode == "mcq":
            question = input("Question: ").strip()
            print("Enter 4 candidates, one per line:")
            candidates = [input(f"{i+1}: ").strip() for i in range(4)]
            label, raw = run_mkrag_mcq(question, candidates)
            print("\nPrediction:", label)
        elif mode == "open":
            question = input("Question: ").strip()
            ans = run_open_query(question)
            print("\nAnswer:\n", ans)
        else:
            print("Invalid mode. Please type 'mcq' or 'open'.")

if __name__ == "__main__":
    run_chat_loop()