from typing import Any, Dict, List
import numpy as np
from openai import OpenAI
import torch

from sentence_transformers import CrossEncoder, SentenceTransformer
import vllm
import vllm.sampling_params
from rag_baseline import ChunkExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NUM_CONTEXT_SENTENCES = 30
NUM_RERANKED_SENTENCES = 15
MAX_CONTEXT_REFERENCES_LENGTH = 4000

AICROWD_SUBMISSION_BATCH_SIZE = 1
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85
SENTENTENCE_TRANSFORMER_BATCH_SIZE = 32

class BetterRagNoReranker:
    def __init__(
        self,
        llm_name="meta-llama/Llama-3.2-1B-Instruct",
        is_server=False,
        vllm_server=None
    ):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE

        self.chunk_extractor = ChunkExtractor()

        # sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # tf-idf for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        if self.is_server:
            self.llm_client = OpenAI(
                api_key="EMPTY",
                base_url = self.vllm_server
            )
        else:
            self.llm = vllm.LLM(
                model=self.llm_name,
                worker_use_ray=True,
                tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
                trust_remote_code=True,
                dtype="half",
                enforce_eager=True,
            )
            self.tokenizer = self.llm.get_tokenizer()

    def get_batch_size(self) -> int:
        return self.batch_size

    def extract_keywords(self, query: str) -> str:
        keyword_system_prompt = "You are an expert at extracting the most important keywords from a query."
        keyword_user_prompt = f"Get the most important keywords in the query: {query} Include no other texts. Do not answer the query."

        chat = [
            {"role": "system", "content": keyword_system_prompt},
            {"role": "user", "content": keyword_user_prompt}
        ]

        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=chat,
                n=1,
                top_p=0.9,
                temperature=0.1,
                max_tokens=50,
            )
            keywords = response.choices[0].message.content.strip()
        else:
            chat = [self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            )]
            responses = self.llm.generate(
                chat,
                vllm.sampling_params(
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    max_tokens=50,
                ),
                use_tqdm=False
            )
            keywords = responses[0].outputs[0].text.strip()

        return keywords

    def keyword_search(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        if len(chunks) == 0:
            return []

        keywords = self.extract_keywords(query)

        all_texts = [keywords] + list(chunks)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)

        query_vector = tfidf_matrix[0]
        chunk_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(
            query_vector,
            chunk_vectors
        ).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        top_chunks = [chunks[i] for i in top_indices]
        return top_chunks

    def vector_search(self, query: str, chunks: List[str], top_k: int) -> List[str]:
        if len(chunks) == 0:
            return []

        query_vector = self.sentence_transformer.encode(
            sentences=query,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
            convert_to_tensor=True
        )
        chunk_vectors = self.sentence_transformer.encode(
            sentences=chunks,
            normalize_embeddings=True,
            batch_size=SENTENTENCE_TRANSFORMER_BATCH_SIZE,
            convert_to_tensor=True
        )

        similarities = cosine_similarity(
            query_vector.unsqueeze(0).cpu(),
            chunk_vectors.cpu()
        ).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        top_chunks = [chunks[i] for i in top_indices]
        return top_chunks

    def merge_chunks(self, vector_chunks: List[str], keyword_chunks: List[str]) -> List[str]:
        """
        Merges and deduplicates chunks obtained from vector and keyword searches
        """
        combined_chunks = vector_chunks + keyword_chunks
        deduplicated = set(combined_chunks)

        return list(deduplicated)

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        batch_interaction_ids = batch["interaction_id"]
        queries = batch["query"]
        batch_search_results = batch["search_results"]
        query_times = batch["query_time"]

        chunks, chunk_interaction_ids = self.chunk_extractor.extract_chunks(
            batch_interaction_ids,
            batch_search_results
        )

        batch_retrieval_results = []
        for _idx, interaction_id in enumerate(batch_interaction_ids):
            query = queries[_idx]

            relevant_chunks_mask = chunk_interaction_ids == interaction_id
            relevant_chunks = chunks[relevant_chunks_mask]

            vector_search_res = self.vector_search(
                query,
                relevant_chunks,
                top_k=NUM_CONTEXT_SENTENCES
            )
            keyword_search_res = self.keyword_search(
                query,
                relevant_chunks,
                top_k=NUM_CONTEXT_SENTENCES
            )

            merged_chunks = self.merge_chunks(vector_search_res, keyword_search_res)
            selected_chunks = np.random.choice(
                merged_chunks,
                size=min(NUM_RERANKED_SENTENCES, len(merged_chunks)),
                replace=False
            )

            batch_retrieval_results.append(selected_chunks)

        formatted_prompts = self.format_prompts(queries, query_times, batch_retrieval_results)

        if self.is_server:
            response = self.llm_client.chat.completions.create(
                model=self.llm_name,
                messages=formatted_prompts[0],
                n=1,
                top_p=0.9,
                temperature=0.1,
                max_tokens=50,
            )
            answers = [response.choices[0].message.content]
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.sampling_params(
                    n=1,
                    top_p=0.9,
                    temperature=0.1,
                    max_tokens=50,
                ),
                use_tqdm=False
            )
            answers = [response.outputs[0].text for response in responses]

        return answers

    def format_prompts(
        self,
        queries: List[str],
        query_times: List[str],
        batch_retrieval_results: List[List[str]]
    ) -> List[str]:
        system_prompt = (
            "Extract the most precise, short answer phrase from the references. "
            "If the answer is NOT in the references, respond with 'I don't know'. "
            "Do NOT explain the reasoning behind your answers."
        )
        formatted_prompts = []

        for _idx, query in enumerate(queries):
            query_time = query_times[_idx]
            retrieval_results = batch_retrieval_results[_idx]

            references = ""
            if len(retrieval_results) > 0:
                references += "# References \n"
                for snippet in retrieval_results:
                    references += f"- {snippet.strip()}\n"

            references = references[:MAX_CONTEXT_REFERENCES_LENGTH]

            user_prompt = (
                f"{references}\n------\n\n"
                f"Using only the references listed above, answer the following question. "
                f"Current Time: {query_time}\n"
                f"Query: {query}\n"
                f"Answer: "
            )

            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if self.is_server:
                formatted_prompts.append(chat)
            else:
                formatted_prompts.append(self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                ))

        return formatted_prompts