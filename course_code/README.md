# BetterRag: A Hybrid Retrieval-Augmented Generation (RAG) Pipeline

- [BetterRag: A Hybrid Retrieval-Augmented Generation (RAG) Pipeline](#betterrag-a-hybrid-retrieval-augmented-generation-rag-pipeline)
    - [Overview](#overview)
    - [Key Components and Features](#key-components-and-features)
        - [1. **Chunk Extraction**](#1-chunk-extraction)
        - [2. **Keyword and Vector Search**](#2-keyword-and-vector-search)
        - [3. **Reranking**](#3-reranking)
        - [4. **Language Model (LLM)**](#4-language-model-llm)
        - [5. **Prompt Engineering**](#5-prompt-engineering)
    - [Pipeline Execution](#pipeline-execution)
        - [1. **Initialization**](#1-initialization)
        - [2. **Batch Query Execution**](#2-batch-query-execution)
    - [Execution Steps](#execution-steps)
    - [Contributors](#contributors)


## Overview

BetterRag is an advanced Retrieval-Augmented Generation (RAG) pipeline designed to retrieve relevant information and generate high-quality responses to user queries. The model leverages multiple methodologies to ensure effective data retrieval, reranking, and answer generation. This README explains the components, pipelines, and execution flow of the `BetterRag` model.

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
- Load required components such as Chunk Extractor, Sentence Transformer, TF-IDF, and Cross-Encoder.

### 2. **Batch Query Execution**
The core execution flow involves:

**Step 0: Chunk Retrieval**
- The Chunk Extractor parses the document, breaking it up into chunks (document snippets).

**Step 1a: Keyword Search**
- Keyword Extraction:
    - Keywords are extracted from the query using the LLM.
- Search Execution:
    - A TF-IDF Vectorizer performs keyword-based searches against preprocessed text chunks.
    - Chunks containing the keywords are obtained from this step.

**Step 1b: Vector Search**
- A SentenceTransformer model encodes the query and text chunks into dense vectors for vector-based semantic search.
- Chunks with semantically similar meanings (based on sentence-embeddings) are obtained from this step.

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

## Execution Steps
*Note: Our group only ran our models using the external vLLM server. They would likely work without it (using the vLLM library), but the following instructions depend on running the vLLM server separately from the python scripts.*

The following steps assume you are in the `./course_code` directory.

- Start the vLLM server
```sh
./start_vllm.sh
```

- Modify `run_generate.sh` and `run_evaluate.sh` to use the desired dataset/models
    - Supported Datasets (added after the `--dataset_path` flag):
        - `"example_data/dev_data.jsonl.bz2"`
        - `"data/crag_task_1_dev_v4_release.jsonl.bz2"`
    - Supported Models (added after the `--model_name` flag):
        - Baselines
            - `"vanilla_baseline"`
            - `"rag_baseline"`
        - Custom
            - `"betterrag"`,
            - `"betterrag_no_keyword_prompt"`,
            - `"betterrag_no_vector_search"`,
            - `"betterrag_no_reranker"`,

- Run the generation pipeline
```sh
./run_generate.sh
```

- Run the evaluation pipeline
```sh
./run_evaluate.sh
```

*Note: For most cases, the generation and evaluation will be run sequentially. For convenience the following command can be used:*
```sh
./run_generate.sh && ./run_evaluate.sh
```

## Contributors
- Developed and maintained by Group 3:
    - Edward Hwang (hwangedward8@ucla.edu)
    - Allen Liang (aliang20@ucla.edu)
    - Seth Reis (sethreis3@ucla.edu)
    - Bangyan Shi (bangyan3@ucla.edu)
    - Michael Srouji (msrouji@ucla.edu)