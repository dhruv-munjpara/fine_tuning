# Session 2: Prompt Engineering vs RAG vs Fine-Tuning
**Module:** Fine-Tuning & Model Customization  
**Student Repository:** [https://github.com/dhruv-munjpara/fine_tuning](https://github.com/dhruv-munjpara/fine_tuning)

---

## 📌 Question 1: Comprehensive Comparison Matrix

| Evaluation Dimension | Prompt Engineering | Retrieval-Augmented Generation (RAG) | Fine-Tuning (Full / PEFT) |
| :--- | :--- | :--- | :--- |
| **Data Freshness** | **Static / Session-only:** Limited to whatever raw text is pasted into the prompt window. | **Dynamic / Real-time:** Instantly incorporates live documents, databases, and APIs without model retraining. | **Static:** Knowledge is frozen at training time. Updating facts requires dataset recreation and retraining. |
| **Hallucination Control** | **Moderate:** The model is prone to confabulating answers if the base LLM is unfamiliar with the topic. | **Superior / High:** Output is strictly constrained to retrieved evidence passages, enabling verifiable citations. | **Moderate to Low:** Model learns task behavior, but can still hallucinate non-existent facts with high confidence. |
| **Setup Cost & Resources** | **Minimal:** No dedicated GPUs or database infrastructure required; only API inference costs. | **Moderate:** Requires embedding models, vector database (Milvus/Pinecone/Chroma), and chunking pipelines. | **High:** Requires curated labeled datasets, GPU compute clusters (even with LoRA), and ML expertise. |
| **Latency per Inference** | **Variable / High:** Lengthy prompts and multiple few-shot examples increase Time-to-First-Token (TTFT). | **Moderate to High:** Overhead of vector search + embedding latency + injecting retrieved tokens into prompt. | **Minimal / Lowest:** Prompts are short and concise; model can be distilled or run on smaller architectures. |
| **Customization Depth** | **Surface Level:** Influences stylistic guidelines, output formats, and immediate reasoning tone. | **Contextual Level:** Injects external domain knowledge without altering model internal representations. | **Deep Architectural Level:** Modifies neural network weights, attention mechanisms, and feature representations. |

---

## 📌 Question 2: RAG Simulation Workflow (IPL Cricket Stats)

The complete Python implementation is located in [`simulate_rag_workflow.py`](file:///d:/DA%26DS/fine_tuning/session_2_prompt_vs_rag_vs_finetuning/simulate_rag_workflow.py) with the knowledge base stored in [`ipl_stats.txt`](file:///d:/DA%26DS/fine_tuning/session_2_prompt_vs_rag_vs_finetuning/ipl_stats.txt).

### Architectural Breakdown:
1. **Document Loading & Chunking:** Loads knowledge categories (franchise titles, Orange Cap records, Purple Cap records, highest totals, and individual milestones) as distinct contextual chunks.
2. **Retriever Mechanism:**
   - Filters common stop-words.
   - Computes query-document token intersections.
   - Applies domain entity boosts (e.g., player names like *Virat Kohli, MS Dhoni*, team acronyms like *CSK, SRH, MI*).
   - Returns top-$k$ relevant chunks sorted by relevance score.
3. **Augmentation:** Fuses the user question and the retrieved context chunks into an augmented prompt:
   ```text
   [SYSTEM INSTRUCTION]: You are a factual IPL cricket analyst. Answer strictly using retrieved context.
   [RETRIEVED CONTEXT]:
   - Chennai Super Kings (CSK) has won 5 IPL titles under MS Dhoni.
   - Mumbai Indians (MI) has won 5 IPL titles under Rohit Sharma.
   [USER QUESTION]: Which team has won 5 IPL titles and who were the captains?
   ```
4. **Generation / Grounded Synthesis:** Outputs a verified, non-hallucinated response anchored entirely in factual context.

---

## 📌 Question 3: Real-World Scenarios from Daily Apps (Zomato, Flipkart, Paytm)

### 1. Zomato: Personalized Culinary Recommendations (Prompt Engineering)
* **Approach:** **Prompt Engineering (Few-Shot System Prompts)**
* **Scenario:** Dynamic customer chat assistant that suggests dinner combos based on immediate user mood and dietary constraints (e.g., *"Find me spicy North Indian vegetarian food under ₹400 in Indiranagar"*).
* **Why it's the best approach:** The user's preferences are conversational and temporary. Passing the user's filters, current cart contents, and recent order history directly in the prompt context allows an off-the-shelf LLM to generate creative, persuasive descriptions without any model retraining or heavy vector storage.

### 2. Flipkart: Real-Time Order Tracking & Policy Q&A (RAG)
* **Approach:** **Retrieval-Augmented Generation (RAG)**
* **Scenario:** Customer support agent answering queries like: *"Can I return this open-box smartphone within 7 days, and what is my refund status on order #OD8237482?"*
* **Why it's the best approach:** Flipkart carries millions of products, each with unique vendor return policies, warranty terms, and live courier tracking states. Fine-tuning cannot handle real-time inventory and delivery statuses. RAG queries live order databases and product policy manuals to fetch the exact clause and provide an audited, hallucination-free answer with refund policy links.

### 3. Paytm: Automated Financial Fraud & Transaction Anomaly Detection (Fine-Tuning)
* **Approach:** **Fine-Tuning (PEFT / LoRA on Small Encoders)**
* **Scenario:** Classifying millions of UPI transaction logs, merchant payment narratives, and SMS payment gateway messages in sub-10 milliseconds to detect fraudulent scams, phishing links, and money laundering patterns.
* **Why it's the best approach:**
  - **Latency:** RAG and massive LLM prompts take hundreds of milliseconds; fraud gateways require instantaneous (<15ms) binary classifications.
  - **Data Privacy & Cost:** Running millions of micro-transactions through third-party proprietary LLM APIs is financially non-viable and violates RBI data localization rules.
  - **Specialized Vocabulary:** A compact local model (e.g., DistilBERT/RoBERTa) fine-tuned on labeled fraud patterns achieves near-perfect classification accuracy at low hosting cost.

---

## 📌 Question 4: Comparison Table

The complete structured spreadsheet is saved as [`prompt_rag_finetune_comparison.csv`](file:///d:/DA%26DS/fine_tuning/session_2_prompt_vs_rag_vs_finetuning/prompt_rag_finetune_comparison.csv).

| Criteria | Prompt Engineering | Retrieval-Augmented Generation (RAG) | Fine-Tuning |
| :--- | :--- | :--- | :--- |
| **Setup Cost** | **$0 – Minimal:** Only prompt creation and LLM API tokens. | **Moderate:** Vector DB instance, embedding model compute, ETL pipeline. | **High:** Dataset annotation, GPU compute (A100/H100), MLOps pipelines. |
| **Time to Deploy** | **Minutes to Hours:** Write prompt, test edge cases, push to production. | **1 to 3 Days:** Build chunker, generate embeddings, connect vector search. | **Weeks:** Data cleaning, hyperparameter search, validation, and safety tuning. |
| **Maintenance Effort** | **Low:** Update prompt guidelines when requirements change. | **Moderate:** Monitor vector indexing, document staleness, and retrieval precision. | **High:** Regular retraining required as domain data drifts or new tasks emerge. |
| **Ideal Use Cases** | Brainstorming, ad copy generation, summarizing conversation threads, quick demos. | Enterprise search, customer support bots, dynamic product catalogs, legal discovery. | Edge model deployment, specialized classification, proprietary code generation, strict JSON syntax. |

---

## 📌 Question 5: Movie Recommendation Chatbot for BookMyShow (ChatGPT Advice & Final Architecture Decision)

### Simulated Prompt to ChatGPT:
> *"I am building a conversational movie recommendation chatbot for BookMyShow that helps users discover movies currently playing in their city, check showtimes, select genres, and book tickets. Should I use Prompt Engineering, RAG, or Fine-Tuning?"*

### ChatGPT Advice Received:
```text
Recommendation: Use a Hybrid RAG + Function Calling Architecture.

1. Why NOT Fine-Tuning Alone:
   - Movie releases, theater showtimes, ticket availability, and seat layouts change every single day or hour.
   - A fine-tuned model's knowledge is static; it would recommend movies that finished their theatrical run weeks ago.

2. Why NOT Prompt Engineering Alone:
   - You cannot fit entire theater schedules, multi-city listings, and live seat matrix data into an LLM context window.

3. Why RAG + API Tool Calling (Winner):
   - Real-Time Grounding: RAG retrieves synopsis, critic reviews, and user ratings from a vector database.
   - Live Integration: Tool/Function calling queries BookMyShow's live cinema API for today's showtimes, pricing, and theater locations (e.g., PVR, INOX, Cinepolis).
   - Conversational Intelligence: The LLM understands natural language user intents ("Suggest a funny romantic movie for a couple in Bandra after 8 PM") and translates them into targeted search filters.
```

### Final Engineering Decision & Architecture:
* **Selected Architecture:** **RAG + API Tool Calling with Lightweight LoRA Style Adaptation**
  1. **User Interaction & Intent Parsing:** An LLM with optimized system prompts extracts user criteria (City: Mumbai, Area: Bandra, Time: Evening, Genre: Rom-Com).
  2. **Tool Calling (Live BookMyShow API):** Directly queries real-time theater schedules and seat occupancy.
  3. **Vector RAG (Movie Metadata & Reviews):** Queries a vector store containing IMDb/BookMyShow critic consensus, parental ratings, and synopsis to answer queries like *"Is this suitable for a 10-year-old child?"*.
  4. **Fine-Tuning Role (Optional / Future):** If running a dedicated on-premise 7B model to reduce API costs, a LoRA adapter can be trained strictly for **JSON tool-calling and conversation dialogue style**, while keeping the data layer strictly in RAG.
