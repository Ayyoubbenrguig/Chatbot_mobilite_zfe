# faiss_index.py

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from config import FAISS_INDEX_TYPE, RELEVANCE_THRESHOLD
from embeddings import get_vertex_embeddings, format_and_embed_metadata
from bigquery_utils import query_bigquery_metadata
from clients import init_clients
import faiss


# Index Embeddings with FAISS
def build_faiss_index(rag_database: Dict[str, Dict]) -> Tuple[Optional[faiss.Index], Optional[List[str]], Optional[Dict[str, str]], Optional[str]]:

    if not rag_database:
        return None, None, None, "No metadata provided to build FAISS index."

    table_ids_list, _, embeddings_list, texts_map, format_embed_error = format_and_embed_metadata(rag_database)

    if format_embed_error:
        return None, None, None, f"Failed during formatting/embedding: {format_embed_error}"
    if not table_ids_list or embeddings_list is None or not texts_map:
         return None, None, None, "Formatting/embedding returned empty or invalid lists/map."

    valid_embeddings = []
    valid_table_ids = []
    valid_texts_map = {}
    failed_count = 0
    original_texts_map = texts_map if texts_map is not None else {}

    for i, emb in enumerate(embeddings_list):
        if i < len(table_ids_list):
            table_id = table_ids_list[i]
            if emb is not None:
                valid_embeddings.append(emb)
                valid_table_ids.append(table_id)
                if table_id in original_texts_map:
                    valid_texts_map[table_id] = original_texts_map[table_id]
                else:
                    logging.error(f"Consistency error: Table ID {table_id} missing from original text map during filtering.")
            else:
                logging.warning(f"Skipping table '{table_id}' due to embedding failure (None value).")
                failed_count += 1
        else:
             logging.error(f"Index mismatch: Embedding list longer than table ID list at index {i}.")
             failed_count += 1 


    if not valid_embeddings:
        err_detail = f" ({format_embed_error})" if format_embed_error else ""
        return None, None, None, f"No valid embeddings available to build FAISS index.{err_detail}"


    try:
        embeddings_np = np.array(valid_embeddings, dtype='float32')
        if embeddings_np.ndim != 2 or embeddings_np.shape[0] == 0:
             raise ValueError(f"Valid embeddings resulted in an invalid NumPy array shape: {embeddings_np.shape}")
        dimension = embeddings_np.shape[1]
    except ValueError as e:
        error_msg = f"Error converting valid embeddings to NumPy array: {e}"
        logging.error(error_msg, exc_info=True)
        return None, None, None, error_msg

    logging.info(f"Embeddings ready for indexing: {embeddings_np.shape[0]} valid vectors, dimension: {dimension}. ({failed_count} embedding failures)")

    logging.info(f"Building FAISS index ({FAISS_INDEX_TYPE.__name__})...")
    try:
        if dimension <= 0:
             return None, None, None, "Cannot build FAISS index with zero dimension."
        index = FAISS_INDEX_TYPE(dimension)
        index.add(embeddings_np)
        logging.info(f"FAISS index built successfully with {index.ntotal} entries.")
        return index, valid_table_ids, valid_texts_map, None
    except Exception as faiss_e:
         error_msg = f"Error adding embeddings to FAISS index: {faiss_e}"
         logging.error(error_msg, exc_info=True)
         return None, None, None, error_msg


# Step 4: Query FAISS Index
def search_faiss_index(query: str, faiss_index_obj: faiss.Index, top_k: int) -> Tuple[Optional[List[float]], Optional[List[int]], Optional[str]]:

    if faiss_index_obj is None: return None, None, "FAISS index object is None."
    if not hasattr(faiss_index_obj, 'ntotal') or faiss_index_obj.ntotal == 0:
        logging.warning("Attempted to search an empty or invalid FAISS index.")
        return [], [], None 

    try:
        logging.debug("Generating query embedding...")
        query_embedding_list, embed_error = get_vertex_embeddings([query])
        if embed_error or not query_embedding_list or query_embedding_list[0] is None:
            error_msg = f"Failed to generate embedding for the query: {embed_error or 'Embedding was None'}"
            logging.error(error_msg)
            return None, None, error_msg

        query_embedding = query_embedding_list[0]
        if not isinstance(query_embedding, list) or not all(isinstance(x, (int, float)) for x in query_embedding):
             error_msg = f"Invalid query embedding format received: {type(query_embedding)}"
             logging.error(error_msg)
             return None, None, error_msg

        query_embedding_np = np.array([query_embedding], dtype='float32')

        if not hasattr(faiss_index_obj, 'd'):
             return None, None, "FAISS index object missing dimension attribute 'd'."
        if query_embedding_np.shape[1] != faiss_index_obj.d:
             error_msg = f"Query embedding dimension ({query_embedding_np.shape[1]}) does not match index dimension ({faiss_index_obj.d})."
             logging.error(error_msg)
             return None, None, error_msg

        logging.debug(f"Searching FAISS index for top {top_k} results...")
        actual_k = min(top_k, faiss_index_obj.ntotal)
        if actual_k <= 0:
            logging.warning(f"Search requested with k={top_k}, but index size is {faiss_index_obj.ntotal} or k<=0. Returning empty results.")
            return [], [], None 

        distances, indices = faiss_index_obj.search(query_embedding_np, k=actual_k)

        return distances[0].tolist(), indices[0].tolist(), None

    except Exception as e:
        error_msg = f"Error searching FAISS index: {e}"
        logging.error(error_msg, exc_info=True)
        return None, None, error_msg


# Relevance Check
def check_relevance_faiss(query: str, faiss_index_obj: faiss.Index) -> Tuple[bool, Optional[str]]:

    similarities, indices, search_error = search_faiss_index(query, faiss_index_obj, top_k=1)

    if search_error: return False, f"Relevance check failed due to search error: {search_error}"
    if similarities is None or indices is None:
        return False, "Relevance check failed: Search returned None unexpectedly."
    if not similarities: 
        logging.info("Relevance check: No similar items found in index.")
        return False, None 
    max_similarity = similarities[0]
    logging.info(f"Max relevance score (FAISS IP): {max_similarity:.4f} (Threshold: {RELEVANCE_THRESHOLD})")
    is_relevant = max_similarity >= RELEVANCE_THRESHOLD
    return is_relevant, None

# Find Relevant Tables
def find_relevant_tables_faiss(query: str, faiss_index_obj: faiss.Index, metadata_table_ids: List[str], metadata_texts_map: Dict[str, str], top_k: int = 3) -> Tuple[Optional[List[str]], Optional[str]]:

    similarities, indices, search_error = search_faiss_index(query, faiss_index_obj, top_k=top_k)

    if search_error: return None, f"Relevant tables search failed: {search_error}"
    if indices is None or similarities is None: 
        return None, "Relevant tables search failed (search returned None unexpectedly)."
    if not indices: 
        logging.info("No relevant tables found via FAISS search.")
        return [], None 

    relevant_metadata_strings = []
    logging.info("Relevant tables based on FAISS search:")
    num_ids = len(metadata_table_ids) 
    for i, idx in enumerate(indices):
        if idx < 0 or idx >= num_ids:
            logging.warning(f"Invalid index {idx} returned from FAISS search at position {i} (max index: {num_ids-1}). Skipping.")
            continue

        similarity = similarities[i]
        if similarity >= RELEVANCE_THRESHOLD:
            table_id = metadata_table_ids[idx]
            if table_id in metadata_texts_map:
                 relevant_metadata_strings.append(metadata_texts_map[table_id])
                 logging.info(f"- {table_id} (Score: {similarity:.4f})")
            else:
                 logging.warning(f"Table ID {table_id} found via FAISS (index {idx}) but its text description is missing from map.")
        else:
             logging.info(f"Stopping relevance check at score {similarity:.4f} (below threshold {RELEVANCE_THRESHOLD})")
             break

    return relevant_metadata_strings, None
