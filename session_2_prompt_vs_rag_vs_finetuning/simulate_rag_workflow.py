"""
Session 2 - Task 2: Simulate a Retrieval-Augmented Generation (RAG) Workflow
Module: FINE-TUNING & MODEL CUSTOMIZATION

Task Requirements:
Simulate a Retrieval-Augmented Generation (RAG) workflow by creating a Python script
that takes a user's question about IPL cricket stats, searches a provided IPL stats text file
for relevant information, and prints a combined answer using both the question and retrieved data.
Hint: Use basic file reading and string matching to find relevant lines in the text file.
"""

import os
import re
from typing import List, Dict, Tuple

class IPLStatsRAGSimulator:
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.documents = []
        self._load_and_chunk_knowledge_base()

    def _load_and_chunk_knowledge_base(self):
        """Loads the knowledge base file and breaks it down into searchable knowledge chunks."""
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Knowledge base file not found at: {self.data_file_path}")

        current_category = "General"
        with open(self.data_file_path, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("["):
                    continue
                if line.startswith("Category:"):
                    current_category = line.replace("Category:", "").strip()
                else:
                    self.documents.append({
                        "id": line_no,
                        "category": current_category,
                        "content": line
                    })
        print(f"Loaded {len(self.documents)} knowledge records from '{os.path.basename(self.data_file_path)}'.")

    def _tokenize(self, text: str) -> set:
        """Tokenize and remove common stop words for clean string matching."""
        stop_words = {
            "what", "is", "the", "in", "a", "an", "who", "which", "how", "many",
            "has", "have", "of", "and", "or", "to", "for", "by", "with", "most",
            "all", "time", "highest", "record", "ipl", "history", "score"
        }
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {t for t in tokens if t not in stop_words and len(t) > 1}

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Retriever component:
        Uses keyword token matching, entity recognition, and relevance scoring.
        """
        query_lower = query.lower()
        query_tokens = self._tokenize(query)
        scored_docs: List[Tuple[float, Dict]] = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            doc_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", content_lower))

            score = 0.0
            # Token overlap score
            overlap = query_tokens.intersection(doc_tokens)
            score += len(overlap) * 2.0

            # Substring / Exact phrase matching boost
            for token in query_tokens:
                if token in content_lower:
                    score += 1.5

            # Key entity boosts (player names, team codes)
            key_entities = [
                "csk", "chennai", "mi", "mumbai", "srh", "hyderabad", "rcb", "bengaluru",
                "kkr", "kolkata", "gt", "gujarat", "rr", "rajasthan", "pbks", "punjab",
                "kohli", "dhoni", "rohit", "gayle", "chahal", "raina", "warner",
                "fastest", "wicket", "run", "total", "boundary", "purple", "orange"
            ]
            for entity in key_entities:
                if entity in query_lower and entity in content_lower:
                    score += 3.0

            if score > 0:
                scored_docs.append((score, doc))

        # Sort descending by relevance score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def generate_augmented_response(self, query: str) -> str:
        """
        Augmentation & Synthesis component:
        Combines user query with retrieved knowledge chunks into an augmented prompt,
        then synthesizes a factual, grounded response.
        """
        retrieved_docs = self.retrieve(query, top_k=2)

        print("\n" + "=" * 70)
        print(f"USER QUERY: \"{query}\"")
        print("=" * 70)

        if not retrieved_docs:
            print("\n[Retrieval Status]: No relevant context found in local knowledge base.")
            return "I do not have sufficient information in the IPL stats database to answer this question."

        print(f"\n[Retrieval Phase] Found {len(retrieved_docs)} relevant context snippet(s):")
        context_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"  [{i}] (Category: {doc['category']}) -> {doc['content']}")
            context_text += f"- {doc['content']}\n"

        # Construct the Augmented Prompt
        augmented_prompt = (
            f"=== AUGMENTED PROMPT (Passed to Generator) ===\n"
            f"[SYSTEM INSTRUCTION]: You are a factual IPL cricket analyst. Answer the user question "
            f"strictly using the verified context provided below. Do not hallucinate.\n\n"
            f"[RETRIEVED CONTEXT]:\n{context_text}\n"
            f"[USER QUESTION]: {query}\n"
            f"================================================"
        )
        print(f"\n{augmented_prompt}")

        # Generation / Synthesis Logic
        # In full production RAG, this prompt is sent to an LLM API.
        # Here we synthesize the grounded answer directly using the retrieved facts:
        synthesis_lines = []
        for doc in retrieved_docs:
            synthesis_lines.append(f"According to verified IPL records: {doc['content']}")

        final_answer = "\n".join(synthesis_lines)
        return final_answer


def run_demo():
    kb_path = os.path.join(os.path.dirname(__file__), "ipl_stats.txt")
    rag = IPLStatsRAGSimulator(kb_path)

    test_queries = [
        "Which team has won 5 IPL titles and who were the captains?",
        "Who is the all-time highest run scorer in IPL history and how many runs?",
        "What is the highest team total in IPL history and against whom was it scored?",
        "Who holds the record for the fastest fifty in IPL history?",
        "Who has taken the most wickets in IPL all-time?"
    ]

    for q in test_queries:
        response = rag.generate_augmented_response(q)
        print("\n[GENERATED GROUNDED ANSWER]:")
        print(response)
        print("-" * 70)


if __name__ == "__main__":
    run_demo()
