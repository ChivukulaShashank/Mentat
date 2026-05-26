# Mentat

Mentat is an offline, Retrieval-Augmented Generation (RAG) system designed for medical document analysis. Inspired by the "human computer" concept, Mentat automates the extraction, indexing, and querying of clinical documentation to provide fast, locally-hosted insights while ensuring data privacy.

## Features
- **Offline-First:** All processing, embedding, and inference happens locally on your machine.
- **Automated Pipeline:** Full end-to-end processing (NER extraction --> FAISS indexing --> LLM Query).
- **Context:** Built to process complex text and provide concise, fact-based answers.
- **Privacy Focused:** No cloud APIs; your data never leaves your local environment.

## Getting Started

### Prerequisites
- Python 3.10+
- [GPT4All](https://gpt4all.io/) model file (`.gguf`) placed in your `/models` folder.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Mentat.git](https://github.com/YOUR_USERNAME/Mentat.git)
   cd Mentat
Install dependencies:

Bash
pip install -r requirements.txt
Place your PDF in the /data folder.

Usage
Run the pipeline to process your documents and launch the interactive chatbot:

Bash
python main.py

System Pipeline & Logic
Mentat follows a modular Extraction-Index-Inference architecture designed to maintain high privacy while minimizing computational overhead.

1. Entity-Aware Ingestion (NER)
The pipeline begins by processing raw PDFs through the MentatNER component. Unlike standard text-splitters, this stage uses a medical-domain token classification model to identify and isolate key clinical entities before text is indexed. This ensures that the search index is populated with high-signal, medically relevant data.

2. Vectorization & Persistent Indexing
Once cleaned, text is embedded into a vector space. We use a locally hosted FAISS index to ensure that search operations remain high-performance and dependency-free.

Why this logic? By separating the Embedding stage from the Chat stage, we ensure the index is only built once, preventing redundant compute cycles during document querying.

3. Retrieval-Augmented Generation (RAG)
When a user submits a query:

Retrieval: The system searches the FAISS index to find the most relevant context snippets based on vector similarity.

Generation: These snippets are injected into a prompt as a constrained "Knowledge Facts" block.

Constraint: The model is explicitly instructed to rely only on the provided knowledge facts, reducing the likelihood of hallucinations while maintaining the conversational flexibility of the Mistral LLM.

4. Design Philosophy: "Lazy Initialization"
The pipeline is intentionally decoupled. Each major component (MentatNER, MentatEmbedder, MentatRetriever) is initialized only when the specific execution phase requires it.

The Benefit: This prevents the system from attempting to load heavy model weights into VRAM/RAM prematurely, allowing the script to run seamlessly on consumer-grade hardware without memory bottlenecks.
