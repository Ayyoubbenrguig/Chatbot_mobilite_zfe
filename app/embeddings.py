# embeddings.py
from typing import List, Tuple, Union, Optional, Dict
import logging
from clients import embedding_model, CLIENTS_INITIALIZED, CLIENT_INIT_ERROR
from config import EMBEDDING_BATCH_SIZE

# Embedding Helper
def get_vertex_embeddings(texts: List[str]) -> Tuple[List[Union[List[float], None]], Optional[str]]:
    if not CLIENTS_INITIALIZED or embedding_model is None:
        return [None] * len(texts), f"Embedding client not initialized: {CLIENT_INIT_ERROR}"

    all_embeddings: List[Union[List[float], None]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        try:
            safe_batch = [str(item) if item is not None else "" for item in batch]
            embeddings_response = embedding_model.get_embeddings(safe_batch)
            all_embeddings.extend([e.values for e in embeddings_response])
        except Exception as e:
            error_msg = f"Error getting embeddings for batch starting at index {i}: {e}"
            logging.error(error_msg, exc_info=True)
            all_embeddings.extend([None] * len(batch))

    if all(e is None for e in all_embeddings) and texts:
         return all_embeddings, "Failed to generate any embeddings."

    return all_embeddings, None


# Step 2: Generate Embeddings for Metadata
def format_and_embed_metadata(rag_database: Dict[str, Dict]) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[Union[List[float], None]]], Optional[Dict[str, str]], Optional[str]]:

    metadata_texts = []
    table_ids = []
    metadata_texts_map = {}

    logging.info("Formatting metadata into text descriptions for embedding...")
    for full_table_id, metadata in rag_database.items():
        if not metadata or "schema" not in metadata:
             logging.warning(f"Skipping embedding for {full_table_id} due to missing metadata/schema.")
             continue
        try:
            schema_str = "\n".join([f"    - `{name}` ({dtype})" for name, dtype in metadata.get('schema', {}).items()])
            unique_str = "\n".join([f"    - `{name}`: {val}" for name, val in metadata.get('unique_values', {}).items()])
            metadata_text = (
                f"Table Name: `{full_table_id}`\n"
                f"Schema:\n{schema_str if schema_str else '    N/A'}\n"
                f"Unique Values/Ranges (Sampled):\n{unique_str if unique_str else '    N/A'}"
            )
            metadata_texts.append(metadata_text)
            table_ids.append(full_table_id)
            metadata_texts_map[full_table_id] = metadata_text
        except Exception as e:
            logging.warning(f"Could not format metadata for {full_table_id}: {e}")
            continue

    if not metadata_texts:
        return None, None, None, None, "No metadata texts could be formatted for embedding."

    logging.info(f"Generating embeddings for {len(metadata_texts)} descriptions...")
    embeddings, embed_error = get_vertex_embeddings(metadata_texts)

    if embed_error:
        logging.error(f"Embedding generation failed: {embed_error}")
        return None, None, None, None, f"Embedding generation failed: {embed_error}"
    if embeddings is not None and all(e is None for e in embeddings):
        return None, None, None, None, "All embedding attempts resulted in None."

    return table_ids, metadata_texts, embeddings, metadata_texts_map, None
