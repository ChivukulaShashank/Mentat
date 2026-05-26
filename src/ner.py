import os
import fitz
import re
import nltk
from dotenv import load_dotenv
from transformers import pipeline

# Centralize configuration
load_dotenv()
nltk.download("punkt", quiet=True)

class MentatNER:
    def __init__(self):
        # Initialize your resources here
        self.ner_pipe = pipeline(
            "token-classification", 
            model=os.getenv("NER_MODEL_ID", "Helios9/BioMed_NER"), 
            aggregation_strategy="simple"
        )

    def extract_text(self, pdf_path: str) -> str:
        """Centralized PDF extraction logic."""
        doc = fitz.open(pdf_path)
        # Use a list comprehension for efficiency
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Utility method to clean text."""
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def process(self, pdf_path: str, output_txt: str = "knowledge_base.txt"):
        """The 'one-click' execution method."""
        raw_text = self.extract_text(pdf_path)
        entities = self.ner_pipe(raw_text)
        
        # Write results
        with open(output_txt, "w", encoding="utf-8") as f:
            for ent in entities:
                f.write(f"{ent['entity_group']}: {ent['word']}\n")
        print(f"Extracted {len(entities)} entities to {output_txt}")