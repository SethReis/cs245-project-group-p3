# BetterRag: A Hybrid Retrieval-Augmented Generation (RAG) Pipeline

## Overview

BetterRag is an advanced Retrieval-Augmented Generation (RAG) pipeline designed to retrieve relevant information and generate high-quality responses to user queries. The model leverages multiple tools and methodologies to ensure effective data retrieval, reranking, and answer generation. This README explains the components, pipelines, and execution flow of the `BetterRag` model.

---

## Key Components and Features

### 1. **Chunk Extraction**
- The `ChunkExtractor` is used to process input documents and split them into smaller, meaningful chunks for retrieval and ranking.

### 2. **Keyword and Vector Search**
- **Keyword Search**: Uses TF-IDF (`sklearn`) for lightweight keyword-based retrieval.
- **Vector Search**: Uses a pretrained SentenceTransformer (`all-MiniLM-L6-v2`) for semantic vector-based retrieval.

### 3. **Reranking**
- A Cross-Encoder (`ms-marco-MiniLM-L-12-v2`) reorders retrieved chunks based on their relevance to the query, ensuring the most pertinent information is prioritized.

### 4. **Language Model (LLM)**
- **Server Mode**: Integrates with OpenAI for generating answers when a server is available.
- **Local Mode**: Utilizes `vllm` for highly parallelized and efficient language model inference.

### 5. **Prompt Engineering**
- Prompts are dynamically generated to guide the language model in extracting concise and accurate answers using retrieved and reranked references.

---

## Pipeline Execution

### 1. **Initialization**
- Configure model and device settings.
- Load required components such as ChunkExtractor, Sentence Transformer, TF-IDF, and Cross-Encoder.

### 2. **Batch Query Execution**
The core execution flow involves:
**Step 1: Keyword and Vector-Based Search**
- Keyword Extraction:
    - Keywords are extracted from the query using the LLM.
    - The extraction process prioritizes the most important terms for searching relevant information.
- Search Execution:
    - A TF-IDF Vectorizer performs keyword-based searches against preprocessed text chunks, prioritizing n-gram matches.
    - Simultaneously, a SentenceTransformer model encodes the query and text chunks into dense vectors for vector-based semantic search.

**Step 2: Merging and Deduplication**
- Results from both keyword-based and vector-based searches are combined into a single list.
- To avoid redundancy, duplicate entries across both search methods are removed, resulting in a clean set of context chunks for the next step.

**Step 3: Reranking Results**
- The merged list is passed to a Cross-Encoder model, which evaluates the relevance of each chunk to the original query.
- The chunks are scored based on relevance, and the top-ranked results are selected for answer generation.

**Step 4: Prompt Formatting and Answer Generation**
- The selected context chunks are formatted into prompts for the LLM.
- Prompts include:
    - References: Snippets from the retrieved context chunks.
    - Query Metadata: The original query and a timestamp for contextual awareness.
- The prompts are processed by the LLM to generate precise, concise answers.

## Future Work
- Integration with external document databases.
- Optimized memory usage for larger LLM models.

## Contributors
- Developed and maintained by Group 3:
    - Edward Hwang
    - Allen Liang
    - Seth Reis
    - Bangyan Shi
    - Michael Srouji