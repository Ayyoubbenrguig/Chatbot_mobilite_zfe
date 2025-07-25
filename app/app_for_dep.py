
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold, Candidate
from pandas.api.types import is_numeric_dtype
import faiss

import logging, sys
import google.auth
from google.cloud import bigquery, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

from typing import Tuple, Dict, Optional
import pandas as pd
import logging
from pandas.api.types import is_numeric_dtype
from google.cloud import bigquery
from typing import List, Tuple, Union, Optional, Dict
import logging

from embeddings import get_vertex_embeddings, format_and_embed_metadata
from bigquery_utils import query_bigquery_metadata
from clients import init_clients
import faiss


import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re, logging

from datetime import datetime
from datetime import datetime
import datetime
import os
import logging
import streamlit as st

from google.cloud import bigquery
import base64

# chat_logic.py
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import streamlit as st
import logging

##_______________________________________________________________config.py________________________________________

# GCP & BigQuery
PROJECT_ID = "irn-74856-zlb-lab-c5"
DATASET_ID = "Data_mobilite"
LOCATION = "europe-west1"
TABLE_PREFIX_FILTER = "France_Table"

# Vertex AI
EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"
MODEL_ID = "gemini-1.5-pro-002"

BQ_DATASET_REF = f"{PROJECT_ID}.{DATASET_ID}"
# Pipeline RAG
MAX_SAMPLE_VALUES = 10
EMBEDDING_BATCH_SIZE = 5
RELEVANCE_THRESHOLD = 0.2
MAX_RESULTS_FOR_CONTEXT = 20
MAX_CONVERSATION_HISTORY = 5

# Inner Product for cosine similarity
FAISS_INDEX_TYPE = faiss.IndexFlatIP

# Generation configs

SQL_GENERATION_CONFIG = GenerationConfig(temperature=0.0, top_p=0.95, max_output_tokens=1024)
ANSWER_GENERATION_CONFIG = GenerationConfig(temperature=0.3, top_p=0.95, max_output_tokens=1024)
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    # … autres catégories …
}

# Safety Settings
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

# UI
APP_TITLE = "Renault Geo-Mobility Explorer"
LOGO_PATH = "logo.png"
BACKGROUND_IMAGE_PATH = "logo 2.jpg"


##------------------------------------------------------client.py-----------------------------------------------------


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

CLIENTS_INITIALIZED = False
CLIENT_INIT_ERROR = None
bq_client = None
embedding_model = None
generative_model = None

def init_clients():
    global CLIENTS_INITIALIZED, CLIENT_INIT_ERROR, bq_client, embedding_model, generative_model
    try:
        google.auth.default()
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        bq_client = bigquery.Client(project=PROJECT_ID)
        embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        generative_model = GenerativeModel(MODEL_ID)
        CLIENTS_INITIALIZED = True
        logging.info("Clients initialized successfully.")
    except Exception as e:
        CLIENTS_INITIALIZED = False
        CLIENT_INIT_ERROR = e
        logging.error(f"Client initialization failed: {e}", exc_info=True)

# Lancer à l’import
init_clients()



##------------------------------------------------------bigquery_utilis.py-------------------------------------------

# Metadata Fetching
def query_bigquery_metadata(dataset_id: str) -> Tuple[Optional[Dict[str, Dict]], Optional[str]]:

    if not CLIENTS_INITIALIZED:
        return None, f"BigQuery client not initialized: {CLIENT_INIT_ERROR}"
    if bq_client is None:
         return None, "BigQuery client is None."

    logging.info(f"Fetching metadata using SELECT * for dataset: {PROJECT_ID}.{dataset_id} (prefix: '{TABLE_PREFIX_FILTER}')")
    table_summaries = {}
    dataset_ref = bq_client.dataset(dataset_id, project=PROJECT_ID)
    tables_to_process = []
    try:
        tables = bq_client.list_tables(dataset_ref)
        tables_to_process = [tbl for tbl in tables if tbl.table_id.startswith(TABLE_PREFIX_FILTER)]
        logging.info(f"Found {len(tables_to_process)} tables with prefix '{TABLE_PREFIX_FILTER}'.")
    except Exception as e:
         error_msg = f"Error listing tables for {PROJECT_ID}.{dataset_id}: {e}"
         logging.error(error_msg, exc_info=True)
         return None, error_msg

    processed_count = 0
    for table in tables_to_process:
        table_name = table.table_id
        full_table_id = f"{PROJECT_ID}.{dataset_id}.{table_name}"
        logging.info(f"Processing table {processed_count + 1}/{len(tables_to_process)}: {full_table_id}")
        try:
            query = f"SELECT * FROM `{full_table_id}`" 
            logging.debug(f"Executing query: {query}")
            df = bq_client.query(query).to_dataframe(progress_bar_type=None)
            
            # Cas où l’échantillon est vide
            if df.empty:
                logging.warning(f"Table {full_table_id} sample is empty. Getting schema only.")
                try:
                    table_ref_for_schema = bq_client.get_table(f"{PROJECT_ID}.{dataset_id}.{table_name}")
                    schema_dict = {field.name: field.field_type for field in table_ref_for_schema.schema}
                except Exception as schema_e:
                     logging.error(f"Could not get schema for empty table {full_table_id}: {schema_e}")
                     schema_dict = {}

                table_summaries[full_table_id] = {
                     "schema": schema_dict,
                     "unique_values": {col_name: "Table appears empty" for col_name in schema_dict} if schema_dict else {},
                     "project_id": PROJECT_ID,
                     "dataset_id": dataset_id,
                     "table_name": table_name
                }
                processed_count += 1
                continue
            # Cas où l’échantillon contient des lignes
            table_summaries[full_table_id] = {
                "schema": {col: str(df[col].dtype) for col in df.columns},
                "unique_values": {},
                "project_id": PROJECT_ID,
                "dataset_id": dataset_id,
                "table_name": table_name
            }

            logging.debug(f"Sampling unique values/ranges for {full_table_id}...")
        
            for col in df.columns:
                if is_numeric_dtype(df[col]):
                    min_val, max_val = df[col].min(), df[col].max()
                    if pd.isna(min_val) and pd.isna(max_val) and df[col].isnull().all():
                         table_summaries[full_table_id]["unique_values"][col] = "All NULL values in sample"
                    elif pd.isna(min_val) and pd.isna(max_val):
                         table_summaries[full_table_id]["unique_values"][col] = "Range calculation failed (check data type)"
                    else:
                         table_summaries[full_table_id]["unique_values"][col] = f"Sample Range: [{min_val}, {max_val}]"
                else:
                    try:
                        unique_vals = df[col].astype(str).dropna().unique().tolist()
                        if not unique_vals:
                            table_summaries[full_table_id]["unique_values"][col] = "All NULL/No unique values in sample"
                        else:
                            display_vals = [str(v) for v in unique_vals[:MAX_SAMPLE_VALUES]]
                            suffix = "..." if len(unique_vals) > MAX_SAMPLE_VALUES else ""
                            table_summaries[full_table_id]["unique_values"][col] = f"Sample Unique: {display_vals}{suffix}"
                    except Exception as unique_err:
                         logging.warning(f"Could not get unique values for column {col} in {full_table_id}: {unique_err}")
                         table_summaries[full_table_id]["unique_values"][col] = "Error fetching unique values"
            processed_count += 1
        except Exception as e:
            logging.error(f"Error processing table {full_table_id}: {e}", exc_info=True)
            continue

    if not table_summaries and tables_to_process:
        return None, f"Processed {len(tables_to_process)} tables but failed to extract metadata from any."
    elif not tables_to_process:
         return {}, None

    logging.info(f"Successfully processed metadata for {len(table_summaries)} tables.")
    return table_summaries, None


# Execute SQL Query:

def execute_sql_query(sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    if not CLIENTS_INITIALIZED or bq_client is None: return None, f"BigQuery client not initialized: {CLIENT_INIT_ERROR}"
    if not sql_query or not isinstance(sql_query, str):
        return None, f"Invalid SQL query provided (type: {type(sql_query)})."

    logging.info("Executing BigQuery query...")
    logging.debug(f"Executing SQL:\n{sql_query}") 
    try:
        query_job = bq_client.query(sql_query)
        results_df = query_job.to_dataframe(progress_bar_type=None)
        logging.info(f"Query executed successfully, {len(results_df)} rows returned.")
        return results_df, None
    except Exception as e:
        bq_error_reason = ""
        error_info = ""
        if hasattr(e, 'errors') and e.errors and isinstance(e.errors, list) and len(e.errors) > 0:
             first_error = e.errors[0]
             if isinstance(first_error, dict):
                 reason = first_error.get('reason', 'Unknown')
                 message = first_error.get('message', 'No message')
                 location = first_error.get('location', '')
                 bq_error_reason = f" Reason: {reason}, Message: {message}"
                 if location:
                     bq_error_reason += f" Location: {location}"
                 error_info = str(e.errors) 

        error_msg = f"Error executing BigQuery query: {e}{bq_error_reason}"
        logging.error(f"{error_msg}\nBQ Errors: {error_info}\n--QUERY START--\n{sql_query}\n--QUERY END--", exc_info=False)
        return None, error_msg
    

##-------------------------------------------------embeddings.py--------------------------------------------------

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

#------------------------------------------------------faiss_index------------------------------------------------

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

##-----------------------------------------generation.py------------------------------------------------------------

# Format History
def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """Formats conversation history for inclusion in prompts."""
    if not history: return "No previous conversation history."
    formatted = [f"{'User' if turn['role'] == 'user' else 'Assistant'}: {str(turn.get('content','')).replace('---', '- - -')}"
                 for turn in history]
    return "\n".join(formatted)

# Step 5: Generate SQL Script
def generate_sql_query_gemini(
    current_query: str,
    relevant_metadata_strings: List[str],
    conversation_history: List[Dict[str, str]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generates BigQuery SQL using Gemini Pro, considering conversation history.
    Returns tuple: (sql_query, error_message)
    """
    if not CLIENTS_INITIALIZED or generative_model is None: return None, f"Generative model not initialized: {CLIENT_INIT_ERROR}"
    if not relevant_metadata_strings: return None, "No relevant table schemas were identified to generate SQL."

    schema_context = "\n\n---\n\n".join(map(str, relevant_metadata_strings)) 
    history_context = format_history_for_prompt(conversation_history)

    prompt = f"""**Objective:** Generate a single, valid BigQuery SQL query based ONLY on the provided table schemas, the current user question, and the conversation history.

**Instructions:**
1.  **Analyze Context:** Use 'Recent Conversation History' and 'Available Table Schemas' to understand the 'Current User Question'.
2.  **Output ONLY SQL:** Generate *only* the SQL query. No explanations, comments, markdown (like ```sql), or intro text.
3.  **Fully Qualified Names:** Use exact table names from schemas (e.g., `{PROJECT_ID}.{DATASET_ID}.table_name`).
4.  **Joins:** Infer JOIN conditions based on common columns if needed. Use standard JOIN syntax. always use the respective weights when the user asks as represented (eg: How many households represented means you should consider the household weight and similarly for car).
5.  **Implicit Rules:** If `co2_emission` is mentioned, assume unit is tonnes. If asked generally without grouping, calculate `SUM`. Assume `co2_emission` calculation for short distance is `SUM(((co2_emission * local_trip_day_weight))*51)/1e6` and use where local_mobility = 1 and for long trip `SUM(co2_emission * annual_weight) / 1e6` (do not multiply by 51). Always include weights (`local_trip_day_weight` or `annual_weight`) if available in the queried table when calculating CO2.
6.  **Impossibility:** If the question cannot be answered using *only* the provided schemas and history, respond with the exact text: `QUERY_NOT_POSSIBLE`

**Recent Conversation History:**
---
{history_context}
---

**Available Table Schemas:**
---
{schema_context}
---

**Current User Question:**
{current_query}

**BigQuery SQL Query:**
"""

    logging.info("Generating SQL query with Gemini (considering history)...")

    try:
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt)}")

        response = generative_model.generate_content(
            prompt,
            generation_config=SQL_GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            stream=False,
        )

        if not response.candidates or not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
            finish_reason_val = 'Unknown'
            safety_ratings_val = 'Unknown'
            blocked_reason_detail = ""
            if response.candidates:
                 candidate = response.candidates[0]
                 finish_reason_val = getattr(candidate, 'finish_reason', 'N/A')
                 safety_ratings_list = getattr(candidate, 'safety_ratings', [])
                 safety_ratings_val = str(safety_ratings_list) 
                 if safety_ratings_list:
                     for rating in safety_ratings_list:
                         if getattr(rating, 'blocked', False):
                             category_name = getattr(rating.category, 'name', 'Unknown')
                             blocked_reason_detail = f" (Blocked by category: {category_name})"
                             break
            else:
                 prompt_feedback = getattr(response, 'prompt_feedback', None)
                 if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
                      blocked_reason_detail = f" (Prompt blocked, reason: {prompt_feedback.block_reason})"


            error_msg = f"Gemini SQL generation response blocked or empty. Finish reason: {finish_reason_val}. Safety: {safety_ratings_val}{blocked_reason_detail}"
            logging.warning(error_msg)
            return None, error_msg

        sql_query = response.text.strip()

        if "QUERY_NOT_POSSIBLE" in sql_query:
            logging.warning("Gemini indicated query is not possible based on provided context.")
            return None, "The model determined the query is not possible with the provided table information."

        sql_query = re.sub(r"^\s*```(?:sql)?\s*", "", sql_query, flags=re.IGNORECASE | re.MULTILINE)
        sql_query = re.sub(r"\s*```\s*$", "", sql_query, flags=re.IGNORECASE | re.MULTILINE).strip()

        if not sql_query: return None, "SQL generation resulted in an empty string after cleanup."

        if not re.match(r"^\s*(SELECT|WITH|ALTER|CREATE|DELETE|DROP|INSERT|UPDATE|MERGE|TRUNCATE)\b", sql_query, re.IGNORECASE | re.MULTILINE):
            logging.warning(f"Generated text does not look like SQL after cleanup: '{sql_query[:100]}...'")
            return None, "Generated response did not appear to be a valid SQL query."

        logging.info(f"Generated SQL query:\n{sql_query}")
        return sql_query, None

    except Exception as e:
        error_msg = f"Error generating SQL query with Gemini: {e}"
        logging.error(error_msg, exc_info=True)
        if "DeadlineExceeded" in str(e):
            error_msg += " (The request may have timed out)."
        elif "ResourceExhausted" in str(e):
             error_msg += " (Quota limits may have been reached)."
        return None, error_msg

# Step 7: Generate Contextual Natural Language Answer
def generate_contextual_answer_gemini(
    current_query: str,
    sql_query: Optional[str],
    query_result_df: Optional[pd.DataFrame],
    sql_generation_error: Optional[str],
    sql_execution_error: Optional[str],
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Generate a natural language answer based on context, considering history and potential errors.
    """
    if not CLIENTS_INITIALIZED or generative_model is None: return f"Sorry, the language model is not available ({CLIENT_INIT_ERROR}). I cannot generate a response."

    logging.info("Generating final contextual answer with Gemini (considering history and errors)...")

    results_summary = ""
    executed_sql_for_prompt = sql_query if sql_query else "N/A"

    if sql_generation_error:
        results_summary = f"Could not generate the SQL query needed for your request. Reason: {sql_generation_error}"
        executed_sql_for_prompt = f"Failed to generate. Reason: {sql_generation_error}"
    elif sql_execution_error:
        results_summary = f"The SQL query execution failed. Reason: {sql_execution_error}"
        if sql_query:
             results_summary += f"\nFailed Query:\n```sql\n{sql_query}\n```"
    elif query_result_df is None and not sql_query and not sql_generation_error:
         results_summary = "No specific SQL query was generated for this request. This might happen if the relevant data tables couldn't be identified confidently based on your question."
         executed_sql_for_prompt = "N/A (Not generated due to table identification issue)"
    elif query_result_df is None: 
        results_summary = "The SQL query execution failed or returned no valid data."
        if sql_query: 
             results_summary += f"\nQuery Attempted:\n```sql\n{sql_query}\n```"
    elif query_result_df.empty:
        results_summary = "The SQL query executed successfully but returned no results for your specific question."
        if sql_query: 
             results_summary += f"\nExecuted Query:\n```sql\n{sql_query}\n```"
    else:
        limited_df = query_result_df.head(MAX_RESULTS_FOR_CONTEXT)
        try:
             results_summary = f"The query returned the following data (showing up to {MAX_RESULTS_FOR_CONTEXT} rows):\n"
             results_summary += limited_df.to_markdown(index=False) 
             if len(query_result_df) > MAX_RESULTS_FOR_CONTEXT:
                  results_summary += f"\n\n(Note: {len(query_result_df) - MAX_RESULTS_FOR_CONTEXT} more rows were returned but are not shown.)"
        except Exception as to_markdown_err:
            logging.warning(f"Error converting DataFrame results to markdown: {to_markdown_err}. Falling back to string representation.")
            try:
                 results_summary = f"Query returned results (showing up to {MAX_RESULTS_FOR_CONTEXT} rows shown as text due to formatting issue):\n```\n"
                 results_summary += limited_df.to_string(index=False, max_rows=MAX_RESULTS_FOR_CONTEXT) + "\n```"
                 if len(query_result_df) > MAX_RESULTS_FOR_CONTEXT:
                     results_summary += f"\n\n(Note: {len(query_result_df) - MAX_RESULTS_FOR_CONTEXT} more rows were returned but are not shown.)"
            except Exception as fallback_err:
                 logging.error(f"Fallback DataFrame to_string also failed: {fallback_err}")
                 results_summary = "Query returned results, but there was an error formatting them for display."

    history_context = format_history_for_prompt(conversation_history)

    prompt = f"""**Context:**
1.  **Recent Conversation History:**
    ---
    {history_context}
    ---
2.  **Current User Question:** "{current_query}"
3.  **Attempted BigQuery SQL Query:**
    ```sql
    {executed_sql_for_prompt}
    ```
4.  **Query Outcome & Results Summary:**
    {results_summary}

**Task:** Based *only* on the information provided above (history, current question, SQL attempt, outcome/summary), formulate a clear, concise, and helpful natural language answer *to the current user question*.

**Instructions for Answering:**
*   Directly address the 'Current User Question'.
*   Use the 'Query Outcome & Results Summary' as the primary source.
*   Refer to 'Recent Conversation History' for context and conversational flow.
*   If the summary shows data, present the findings clearly. Format numerical results nicely
*   If the summary indicates 'no results', state that clearly in the context of the question.
*   If the summary indicates a SQL 'generation failure' or 'execution failure', inform the user politely about the issue preventing the data retrieval for their *current request*. Do not apologize excessively. Explain the reason briefly if provided in the summary.
*   If no SQL was generated (e.g., relevance issue), explain that you couldn't pinpoint the data needed based on the available table information.
*   **Do not** invent data or information not present in the summary.
*   **Do not** suggest alternative queries unless the summary explicitly implies ambiguity resolvable by user clarification.
*   **Do not** include instructions/code for saving files.
*   Maintain a helpful, professional tone.

**Answer:**
"""

    try:
        if not isinstance(prompt, str):
             raise TypeError(f"Prompt must be a string, got {type(prompt)}")

        response = generative_model.generate_content(
            prompt,
            generation_config=ANSWER_GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
            stream=False,
        )

        if not response.candidates or not hasattr(response.candidates[0], 'content') or not hasattr(response.candidates[0].content, 'parts') or not response.candidates[0].content.parts:
             finish_reason_val = 'Unknown'
             safety_ratings_val = 'Unknown'
             blocked_reason_detail = ""
             if response.candidates:
                 candidate = response.candidates[0]
                 finish_reason_val = getattr(candidate, 'finish_reason', 'N/A')
                 safety_ratings_list = getattr(candidate, 'safety_ratings', [])
                 safety_ratings_val = str(safety_ratings_list)
                 if safety_ratings_list:
                     for rating in safety_ratings_list:
                         if getattr(rating, 'blocked', False):
                             category_name = getattr(rating.category, 'name', 'Unknown')
                             blocked_reason_detail = f" (Blocked by category: {category_name})"
                             break
             else:
                  prompt_feedback = getattr(response, 'prompt_feedback', None)
                  if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
                       blocked_reason_detail = f" (Prompt blocked, reason: {prompt_feedback.block_reason})"

             logging.warning(f"Gemini answer generation response blocked or empty. Finish reason: {finish_reason_val}{blocked_reason_detail}. Safety: {safety_ratings_val}")
             return f"I received an empty or blocked response while trying to formulate the final answer (Reason: {finish_reason_val}{blocked_reason_detail}). I cannot provide a specific response at this time."

        answer = response.text.strip().strip('`').strip()

        if not answer: 
            logging.warning(f"Gemini answer generation resulted in empty text. Finish reason: {getattr(response.candidates[0], 'finish_reason', 'N/A')}")
            return "I was unable to formulate a response based on the information. This might be due to limitations or the nature of the query outcome."


        logging.info("Generated final answer.")
        return answer

    except Exception as e:
        error_msg = f"Error generating final answer with Gemini: {e}"
        logging.error(error_msg, exc_info=True)
        if "DeadlineExceeded" in str(e):
            error_msg += " (The request may have timed out)."
        elif "ResourceExhausted" in str(e):
             error_msg += " (Quota limits may have been reached)."
        return "I encountered an error while trying to formulate the final answer based on the available information."
    
##-----------------------------------------------------------------feedback.py-------------------------------------

BQ_CLIENT = bigquery.Client()
BQ_DATASET_REF = os.getenv("BQ_DATASET_REF", "irn-74856-zlb-lab-c5.Data_mobilite")

FEEDBACK_TABLE = f"{BQ_DATASET_REF}.Chat_Feedback_list"

# --- Feedback Logging Function ---
def log_feedback(message_id: str, user_query: str, sql_script: str, assistant_response: str, feedback: str):
   
    timestamp = datetime.datetime.now().isoformat()

    row = {
        "timestamp": timestamp,
        "message_id": message_id,
        "user_query": user_query,
        "sql_script" : sql_script,
        "assistant_response": assistant_response,
        "feedback": feedback,
    }

    # 2) Write it to BigQuery
    try:
        errors = BQ_CLIENT.insert_rows_json(FEEDBACK_TABLE, [row])
        if errors:
            logging.error("Failed to insert feedback row: %s", errors)
        else:
            logging.info("Feedback persisted to BigQuery table %s", FEEDBACK_TABLE)
    except Exception as e:
        logging.error("Exception while writing feedback to BigQuery: %s", e, exc_info=True)

    # 3) Also print locally for debugging
    logging.info(f"Feedback logged for message {message_id}: {feedback}")
    print("FEEDBACK RECEIVED:", row)
    
##----------------------------------ui_helpes-------------------------------------------------------------
@st.cache_data 
def get_base64_of_bin_file(bin_file):
    """ Reads a binary file and returns its base64 encoded string. """
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"❌ Asset file not found: {bin_file}")
        return None
    except Exception as e:
        st.error(f"❌ Error reading file {bin_file}: {e}")
        return None


# --- *** UPDATED CSS Section *** ---
def set_page_background_and_styles(png_file):
    """ Sets the background image and custom styles for the Streamlit app. """
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        user_content_selector = 'div[data-testid="stChatMessageContent"][aria-label="Chat message from user"]'

        page_bg_img = f'''
        <style>
        /* Background */
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Make main chat area slightly transparent white to improve text readability */
        [data-testid="stAppViewContainer"] > .main > div:nth-of-type(1) {{
             background-color: rgba(59, 91, 219, 0.85); /* Very subtle white overlay */
             backdrop-filter: blur(2px); /* Minimal blur */
             padding: 10px 20px; /* Add some padding */
             border-radius: 10px;
             margin: 10px; /* Margin around the main content area */
        }}

        /* Input Bar - More neutral */
        .stChatFloatingInputContainer {{
             background-color: rgba(80, 80, 80, 0.7); /* Greyish */
             backdrop-filter: blur(8px);
             border-radius: 10px;
             margin: 10px;
             box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
             border: 1px solid rgba(255, 255, 255, 0.2); /* Subtle border */
        }}
        /* Input text color */
        .stChatInput input {{
            color: white; /* Make input text white */
        }}
        /* Placeholder text color */
        .stChatInput input::placeholder {{
            color: #cccccc; /* Lighter grey for placeholder */
        }}

         /* Chat Messages General */
         div[data-testid="stChatMessage"] {{
             border-radius: 15px;
             /* Add more padding at the bottom to make space for absolute positioned feedback */
             padding: 12px 18px 30px 18px; /* Increased bottom padding */
             margin-bottom: 12px;
             max-width: 85%; /* Set max width */
             box-shadow: 0px 2px 5px rgba(0,0,0,0.15);
             border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
             word-wrap: break-word;
             overflow-wrap: break-word;
             position: relative; /* Crucial for absolute positioning of children */
             clear: both; /* Ensure messages don't overlap horizontally */
             min-height: 50px; /* Ensure minimum height for feedback positioning */
         }}

         /* === Assistant Messages Styles (LEFT ALIGNED, GREEN) === */
         /* Selects messages NOT containing the user message content aria-label */
         div[data-testid="stChatMessage"]:not(:has({user_content_selector})) {{
             background-color: rgba(59, 91, 219, 0.85); /* Darker, less saturated Green */
             margin-left: 0px; /* Align left */
             margin-right: auto; /* Push right boundary */
             float: left; /* Float left */
         }}
         /* Target the text PARAGRAPH inside the assistant message's markdown container */
         div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] p,
         div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] li,
         div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] table,
         div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] pre code {{
             color: white !important;
         }}
         /* Style table headers and cells in assistant messages */
          div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] th,
          div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] td {{
             color: white !important;
             border: 1px solid rgba(255, 255, 255, 0.3);
          }}
         /* Style code blocks background */
         div[data-testid="stChatMessage"]:not(:has({user_content_selector})) div[data-testid="stMarkdownContainer"] pre {{
             background-color: rgba(0, 0, 0, 0.3);
             padding: 8px;
             border-radius: 5px;
         }}

         /* === User Messages Styles (RIGHT ALIGNED, ORANGE) === */
         /* Selects messages containing the user message content aria-label */
         div[data-testid="stChatMessage"]:has({user_content_selector}) {{
             background-color: rgba(229, 238, 255, 0.85); /* Vibrant Orange */
             margin-right: 0px; /* Align right */
             margin-left: auto; /* Push left boundary */
             float: right; /* Float right */
         }}
         /* Target the text PARAGRAPH inside the user message's markdown container */
          div[data-testid="stChatMessage"]:has({user_content_selector}) div[data-testid="stMarkdownContainer"] p {{
             color: black !important;
             text-align: left; /* Keep text left-aligned within bubble */
         }}

         /* Container holding the messages */
         div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {{
             overflow: auto; /* To contain the floated elements */
             padding-bottom: 70px; /* Ensure space above input bar */
         }}


         /* Sidebar */
         [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(0, 0, 0, 0.1);
         }}
          .sidebar-logo {{
              display: block; margin-left: auto; margin-right: auto; margin-bottom: 20px;
          }}
         /* Change sidebar text color */
         section[data-testid="stSidebar"] * {{
            color: #333366 !important;
         }}
         /* Logo animation */
         @keyframes pulse {{
             0% {{ transform: scale(1); opacity: 1; }}
             50% {{ transform: scale(1.05); opacity: 0.85; }}
             100% {{ transform: scale(1); opacity: 1; }}
         }}
         #sidebar-logo-img.pulsing {{
             animation: pulse 1.5s infinite ease-in-out !important;
         }}

         /* Feedback Buttons Container Styling */
         .feedback-container {{
             position: absolute; /* Position relative to the parent stChatMessage */
             bottom: 5px;       /* Position 5px from the bottom */
             right: 10px;       /* Position 10px from the right */
             display: flex;     /* Use flexbox for alignment */
             gap: 5px;          /* Space between buttons */
         }}
        .feedback-container button {{
             background-color: rgba(255, 255, 255, 0.2);
             color: white;
             border: 1px solid rgba(255, 255, 255, 0.5);
             border-radius: 5px;
             padding: 1px 6px; /* Fine-tune padding */
             cursor: pointer;
             transition: background-color 0.2s ease;
             font-size: 0.8em;
             line-height: 1.2; /* Adjust line height for button text/emoji */
         }}
         .feedback-container button:hover {{
             background-color: rgba(255, 255, 255, 0.4);
         }}
         /* Feedback Thank you text */
         .feedback-thanks {{
             position: absolute; /* Position relative to the parent stChatMessage */
             bottom: 5px;       /* Position 5px from the bottom */
             right: 10px;       /* Position 10px from the right */
             font-size: 0.8em;
             color: #e0e0e0;
             background: rgba(0,0,0,0.2); /* Slight background */
             padding: 2px 5px;
             border-radius: 3px;
             line-height: 1.2;
         }}


         /* FAQ Expander Styling */
          div[data-testid="stExpander"] details {{
              border: 1px solid rgba(255, 255, 255, 0.3);
              border-radius: 8px;
              background-color: rgba(80, 80, 80, 0.5);
              margin-bottom: 15px;
          }}
          div[data-testid="stExpander"] summary {{
              color: white;
              font-weight: bold;
          }}
          div[data-testid="stExpander"] p,
          div[data-testid="stExpander"] li {{
              color: #dddddd;
          }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
# --- End of UPDATED CSS Section ---

@st.cache_resource(show_spinner="Initializing Chatbot Engine (Fetching data & building index)...")
def initialize_chatbot_engine():
    """
    Fetches metadata, builds the FAISS index using backend functions defined above.
    Returns tuple: (faiss_index, metadata_table_ids, metadata_texts_map, error_message)
    """
    if not CLIENTS_INITIALIZED:
        err = f"Backend clients failed to initialize: {CLIENT_INIT_ERROR}"
        logging.error(err)
        return None, None, None, err

    # Crée un conteneur vide pour afficher l'état
    status = st.empty()

    # Étape 1
    status.info("Step 1/2: Fetching BigQuery metadata...")
    rag_database_dict, metadata_error = query_bigquery_metadata(DATASET_ID)
    if metadata_error:
        logging.error(f"Metadata Fetching Failed: {metadata_error}")
        status.error(f"❌ Metadata Fetching Failed: {metadata_error}")
        return None, None, None, f"Metadata Fetching Failed: {metadata_error}"

    if rag_database_dict is None:
        msg = "❌ Metadata fetching returned None unexpectedly."
        status.error(msg)
        return None, None, None, msg

    if not rag_database_dict:
        warning_msg = (
            f"⚠️ No tables found matching prefix '{TABLE_PREFIX_FILTER}' "
            f"in dataset '{DATASET_ID}'. Index will be empty."
        )
        logging.warning(warning_msg)
        status.warning(warning_msg)

    # Étape 2
    status.info(f"Found metadata for {len(rag_database_dict)} tables. "
                "Step 2/2: Building FAISS index (includes embedding)...")
    faiss_index_obj, table_ids, texts_map, index_error = build_faiss_index(rag_database_dict)

    if index_error:
        if "No metadata provided" in index_error and not rag_database_dict:
            warning_msg = (
                f"⚠️ No tables found matching prefix '{TABLE_PREFIX_FILTER}'. "
                "Index is empty."
            )
            logging.warning(warning_msg)
            status.warning(warning_msg)
            # crée un index vide de la dimension par défaut
            dimension = 768
            try:
                if embedding_model:
                    dummy_emb = embedding_model.get_embeddings(["test"])
                    if dummy_emb:
                        dimension = len(dummy_emb[0].values)
            except Exception:
                pass
            empty_index = faiss.IndexFlatIP(dimension)
            return empty_index, [], {}, warning_msg
        else:
            logging.error(f"FAISS Index Building Failed: {index_error}")
            status.error(f"❌ FAISS Index Building Failed: {index_error}")
            return None, None, None, f"FAISS Index Building Failed: {index_error}"

    # Vérifications finales
    if faiss_index_obj is None or not hasattr(faiss_index_obj, 'ntotal'):
        err_msg = "Index building returned invalid object unexpectedly."
        logging.error(err_msg)
        status.error(f"❌ {err_msg}")
        return None, None, None, err_msg

    if faiss_index_obj.ntotal == 0 or not table_ids or not texts_map:
        warn_msg = (
            "Index building completed, but the index is empty "
            "(possibly due to embedding failures or no matching tables found)."
        )
        logging.warning(warn_msg)
        status.warning(f"⚠️ {warn_msg}")
        return faiss_index_obj, table_ids, texts_map, warn_msg

    # Succès
    num_indexed = faiss_index_obj.ntotal
    success_msg = f"✅ Chatbot Engine Initialized! ({num_indexed} tables indexed)"
    logging.info(success_msg)
    status.success(success_msg)

    return faiss_index_obj, table_ids, texts_map, None

##--------------------------------------------------chat_logique.py-----------------------------------------------

@st.cache_resource(show_spinner="Initializing Chatbot Engine (Fetching data & building index)...")
def initialize_chatbot_engine():
    """
    Fetches metadata, builds the FAISS index using backend functions defined above.
    Returns tuple: (faiss_index, metadata_table_ids, metadata_texts_map, error_message)
    """
    if not CLIENTS_INITIALIZED:
        err = f"Backend clients failed to initialize: {CLIENT_INIT_ERROR}"
        logging.error(err)
        return None, None, None, err

    # Crée un conteneur vide pour afficher l'état
    status = st.empty()

    # Étape 1
    status.info("Step 1/2: Fetching BigQuery metadata...")
    rag_database_dict, metadata_error = query_bigquery_metadata(DATASET_ID)
    if metadata_error:
        logging.error(f"Metadata Fetching Failed: {metadata_error}")
        status.error(f"❌ Metadata Fetching Failed: {metadata_error}")
        return None, None, None, f"Metadata Fetching Failed: {metadata_error}"

    if rag_database_dict is None:
        msg = "❌ Metadata fetching returned None unexpectedly."
        status.error(msg)
        return None, None, None, msg

    if not rag_database_dict:
        warning_msg = (
            f"⚠️ No tables found matching prefix '{TABLE_PREFIX_FILTER}' "
            f"in dataset '{DATASET_ID}'. Index will be empty."
        )
        logging.warning(warning_msg)
        status.warning(warning_msg)

    # Étape 2
    status.info(f"Found metadata for {len(rag_database_dict)} tables. "
                "Step 2/2: Building FAISS index (includes embedding)...")
    faiss_index_obj, table_ids, texts_map, index_error = build_faiss_index(rag_database_dict)

    if index_error:
        if "No metadata provided" in index_error and not rag_database_dict:
            warning_msg = (
                f"⚠️ No tables found matching prefix '{TABLE_PREFIX_FILTER}'. "
                "Index is empty."
            )
            logging.warning(warning_msg)
            status.warning(warning_msg)
            # crée un index vide de la dimension par défaut
            dimension = 768
            try:
                if embedding_model:
                    dummy_emb = embedding_model.get_embeddings(["test"])
                    if dummy_emb:
                        dimension = len(dummy_emb[0].values)
            except Exception:
                pass
            empty_index = faiss.IndexFlatIP(dimension)
            return empty_index, [], {}, warning_msg
        else:
            logging.error(f"FAISS Index Building Failed: {index_error}")
            status.error(f"❌ FAISS Index Building Failed: {index_error}")
            return None, None, None, f"FAISS Index Building Failed: {index_error}"

    # Vérifications finales
    if faiss_index_obj is None or not hasattr(faiss_index_obj, 'ntotal'):
        err_msg = "Index building returned invalid object unexpectedly."
        logging.error(err_msg)
        status.error(f"❌ {err_msg}")
        return None, None, None, err_msg

    if faiss_index_obj.ntotal == 0 or not table_ids or not texts_map:
        warn_msg = (
            "Index building completed, but the index is empty "
            "(possibly due to embedding failures or no matching tables found)."
        )
        logging.warning(warn_msg)
        status.warning(f"⚠️ {warn_msg}")
        return faiss_index_obj, table_ids, texts_map, warn_msg

    # Succès
    num_indexed = faiss_index_obj.ntotal
    success_msg = f"✅ Chatbot Engine Initialized! ({num_indexed} tables indexed)"
    logging.info(success_msg)
    status.success(success_msg)

    return faiss_index_obj, table_ids, texts_map, None


# --- Main Chatbot Interaction Logic Function ---
def process_chat_query(
    user_query: str,
    chat_history: List[Dict[str, str]]
) -> str:
    """Handles a user query using the backend logic and initialized components from session state."""
    start_time = time.time()
    if "engine_initialized" not in st.session_state or not st.session_state.engine_initialized:
        logging.error("Attempted to process query before engine was initialized.")
        return "Error: Chatbot engine is not initialized. Please wait or refresh."

    faiss_index = st.session_state.get("faiss_index")
    table_ids = st.session_state.get("table_ids")
    texts_map = st.session_state.get("texts_map")

    if faiss_index is None or not hasattr(faiss_index, 'ntotal') or not hasattr(faiss_index, 'd'):
        logging.error("Chatbot FAISS index missing or invalid in session state.")
        return "Sorry, the chatbot's data index is missing or invalid. Initialization might have failed. Please try refreshing."
    if table_ids is None or texts_map is None:
         logging.error("Chatbot metadata components (table_ids, texts_map) missing from session state.")
         return "Sorry, the chatbot's metadata components are missing. Initialization might have failed. Please try refreshing."

    if faiss_index.ntotal == 0:
         logging.warning("FAISS index has 0 entries. Cannot perform RAG.")
         return "My knowledge base seems empty based on the current setup. I cannot answer questions requiring data lookup."

    final_answer = "Sorry, I encountered an issue processing your request."
    sql_script: Optional[str] = None
    query_result_df: Optional[pd.DataFrame] = None
    sql_generation_error: Optional[str] = None
    sql_execution_error: Optional[str] = None
    relevance_error: Optional[str] = None
    find_tables_error: Optional[str] = None
    status_msg = None # Initialize status_msg

    try:
        logging.info(f"Processing query: '{user_query}'")
        status_msg = st.info("Checking relevance and finding related data...", icon="🔍")

        is_relevant, relevance_error = check_relevance_faiss(user_query, faiss_index)

        if relevance_error:
             logging.error(f"Relevance check failed: {relevance_error}")
             sql_generation_error = f"There was an issue checking if your question relates to the available data: {relevance_error}"
             status_msg.error(f"Relevance check failed: {relevance_error}", icon="❌")
        elif is_relevant:
            logging.info("Query determined to be RAG-related.")
            status_msg.info("Finding relevant tables...", icon="📚")
            relevant_metadata_strings, find_tables_error = find_relevant_tables_faiss(
                user_query, faiss_index, table_ids, texts_map, top_k=3
            )

            if find_tables_error:
                logging.error(f"Finding relevant tables failed: {find_tables_error}")
                sql_generation_error = f"Could not find relevant tables: {find_tables_error}"
                status_msg.error(f"Table search failed: {find_tables_error}", icon="❌")
            elif relevant_metadata_strings:
                status_msg.info("Generating SQL query...", icon="⚙️")
                sql_script, sql_generation_error = generate_sql_query_gemini(
                    current_query=user_query,
                    relevant_metadata_strings=relevant_metadata_strings,
                    conversation_history=chat_history
                )
                if sql_generation_error:
                    logging.warning(f"SQL generation failed: {sql_generation_error}")
                    status_msg.warning(f"SQL Generation Issue: {sql_generation_error}", icon="⚠️") 
                elif sql_script:
                    logging.info("Generated SQL, querying BigQuery...")
                    status_msg.info("Executing query on BigQuery...", icon="📊")
                    query_result_df, sql_execution_error = execute_sql_query(sql_script)
                    if sql_execution_error:
                         logging.error(f"SQL execution failed: {sql_execution_error}")
                         status_msg.error(f"Query Execution Failed: {sql_execution_error}", icon="❌") 
                    else:
                         logging.info("Query executed, generating final answer...")
                         status_msg.info("Formatting the answer...", icon="✍️") 

            else: 
                 logging.warning("Query seemed relevant, but couldn't find specific tables above threshold.")
                 sql_generation_error = "While your query seems related, I couldn't identify specific data tables with enough confidence to proceed."
                 status_msg.warning("Could not pinpoint specific tables.", icon="⚠️") # Keep UI feedback
        else:
            logging.info("Query determined to be NOT RAG-related.")
            status_msg.warning("Question doesn't seem related to the indexed data.", icon="❓")
            final_answer = "Sorry, your question doesn't seem related to the data I have access to in the indexed BigQuery tables. I can only answer questions about that specific data."
            status_msg.empty() 
            return final_answer, 

        final_answer = generate_contextual_answer_gemini(
            current_query=user_query,
            sql_query=sql_script,
            query_result_df=query_result_df,
            sql_generation_error=sql_generation_error,
            sql_execution_error=sql_execution_error,
            conversation_history=chat_history
        )
        if status_msg: status_msg.empty() 
        return final_answer, sql_script                       ####### ajout de sql_script

    except Exception as e:
        logging.error(f"An unexpected error occurred processing chat query: {e}", exc_info=True)
        if status_msg: status_msg.error("An unexpected error occurred.", icon="💥")
        return f"An unexpected error occurred while processing your request: {e}. Please try rephrasing or contact support if the issue persists."
    finally:
        end_time = time.time()
        logging.info(f"Query processing took {end_time - start_time:.2f} seconds.")





####################################################------- streamlit run app.py--------------####################

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon=LOGO_PATH)

    set_page_background_and_styles(BACKGROUND_IMAGE_PATH)

    # --- Sidebar ---
    with st.sidebar:
        logo_base64 = get_base64_of_bin_file(LOGO_PATH)
        if logo_base64:
            st.markdown(
                f'<img src="data:image/png;base64,{logo_base64}" class="sidebar-logo" id="sidebar-logo-img" width="100">',
                unsafe_allow_html=True
            )
        st.title(APP_TITLE)
        st.markdown("---")

        # Dataset Overview
        st.markdown("### 📊 Dataset Overview")
        st.markdown("""
            **Scope:** Geographic coverage of all French communes, départements, régions, 
            *bassins de vie*, arrondissements, plus special‐zone tables (ZFE), and records 
            of commune *fusions* & *scissions* up to 2024.

            **Mobility parc:** National‐ and commune‐level stock of passenger (VP) and light‐
            utility (VUL) vehicles from 2011 to 2022, broken down by:
            - Age of vehicle  
            - Fuel type  
            - Vignette category  
            - Economic sector

            **Totals:** ~3.5 million rows across 19 tables.  
        """)

        # At the point where you want to offer the PDF:
        pdf_path = "context_chatbot.pdf"  # or the full path if elsewhere
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📄 Download Full Dataset Context Report (PDF)",
                data=pdf_bytes,
                file_name="Mobilite_Geography_France_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.warning("Context report PDF not found. Make sure it’s generated at startup.")
        
        # Data Source
        st.markdown("### 🔗 Primary Source")
        st.markdown("""
        This data is sourced from INSEE’s open data portal,  
        which provides official statistics on French territorial  
        and socioeconomic indicators.
        """)
        st.markdown(
            "[🇫🇷 View on INSEE](https://www.insee.fr/en/accueil)", 
            unsafe_allow_html=True
        )

        st.markdown("---") 
        if CLIENTS_INITIALIZED:
            st.caption(f"Model: {MODEL_ID}")
            st.caption(f"Dataset: {BQ_DATASET_REF}")
            st.caption(f"Table Prefix: '{TABLE_PREFIX_FILTER}'")
        else:
            st.caption("Backend Clients Unavailable")
            init_err = st.session_state.get("init_error", CLIENT_INIT_ERROR)
            if init_err:
                st.error(f"Client Init Error: {init_err}")
        st.divider()
        st.info("Ask questions about French mobility data")
        sidebar_status_placeholder = st.empty()

    # --- Initialize Session State Variables ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! Please wait while I initialize the connection to the data...", "id": "init_0"}]
    if "engine_initialized" not in st.session_state:
        st.session_state.engine_initialized = False
    if "init_error" not in st.session_state:
        st.session_state.init_error = None 
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "table_ids" not in st.session_state:
        st.session_state.table_ids = None
    if "texts_map" not in st.session_state:
        st.session_state.texts_map = None
    if "message_id_counter" not in st.session_state:
        st.session_state.message_id_counter = 1 
    if "feedback_given" not in st.session_state:
        st.session_state.feedback_given = {} 

    # --- Initialize Backend Engine  ---
    if CLIENTS_INITIALIZED and not st.session_state.engine_initialized and st.session_state.get("init_error") is None:
        faiss_index_obj, table_ids, texts_map, init_msg = initialize_chatbot_engine()

        initial_message_id = st.session_state.messages[0].get('id', 'unknown')

        if init_msg and faiss_index_obj is None: 
            st.session_state.init_error = init_msg 
            st.session_state.engine_initialized = False 
            if initial_message_id == "init_0":
                st.session_state.messages = [{"role": "assistant", "content": f"Initialization Failed: {init_msg}. Cannot proceed. Please check logs or configuration.", "id": "init_error"}]
                st.rerun() 
        elif faiss_index_obj is not None: 
            st.session_state.engine_initialized = True
            st.session_state.faiss_index = faiss_index_obj
            st.session_state.table_ids = table_ids
            st.session_state.texts_map = texts_map
            st.session_state.init_error = init_msg if init_msg else None 

            if initial_message_id == "init_0":
                welcome_msg = "Hi there! I'm ready. How can I help you analyze the data today?"
                is_empty_index = not hasattr(faiss_index_obj, 'ntotal') or faiss_index_obj.ntotal == 0
                if init_msg and "index is empty" in init_msg:
                    welcome_msg += f" (Note: {init_msg})"
                elif is_empty_index:
                    welcome_msg += " (Note: My knowledge base appears empty based on the current configuration.)"
                st.session_state.messages = [{"role": "assistant", "content": welcome_msg, "id": "init_ready"}]
                st.rerun() 


    CHAT_DISABLED = not st.session_state.get("engine_initialized") or not CLIENTS_INITIALIZED or (st.session_state.get("init_error") is not None and st.session_state.get("faiss_index") is None)

    message_container = st.container()
    with message_container:
        for i, message in enumerate(st.session_state.messages):
            sql_for_message = message.get("sql", "")
            msg_id = message.get("id", f"msg_{i}")
            avatar_icon = "🧑" if message["role"] == 'user' else "🤖"
            with st.chat_message(message["role"], avatar=avatar_icon):
                st.markdown(message["content"]) 

                # --- Feedback Section (Rendered using columns within chat_message) ---
                if message["role"] == "assistant" and msg_id not in ["init_0", "init_error", "init_ready", "init_warn"]:
                    feedback_key_base = f"feedback_{msg_id}"
                    feedback_already_given = st.session_state.feedback_given.get(msg_id)

                    current_user_query = "N/A"
                    if i > 0:
                        for j in range(i - 1, -1, -1):
                            if st.session_state.messages[j]["role"] == "user":
                                current_user_query = st.session_state.messages[j]["content"]
                                break

                    feedback_placeholder = st.empty() 

                    if feedback_already_given:
                        feedback_placeholder.markdown(f"<div class='feedback-thanks'>Feedback: {feedback_already_given}</div>", unsafe_allow_html=True)
                    else:
                        with feedback_placeholder.container():
                            st.markdown("<div class='feedback-container'>", unsafe_allow_html=True) # Start container div
                            col_up, col_down = st.columns([1,1]) # Adjust ratios if needed
                            with col_up:
                                if st.button("👍", key=f"{feedback_key_base}_up", help="Mark as helpful"):
                                    st.session_state.feedback_given[msg_id] = "👍"
                                    log_feedback(msg_id, current_user_query, sql_for_message, message["content"], "👍")
                                    st.rerun() # Rerun to update display
                            with col_down:
                                if st.button("👎", key=f"{feedback_key_base}_down", help="Mark as not helpful"):
                                    st.session_state.feedback_given[msg_id] = "👎"
                                    log_feedback(msg_id, current_user_query, sql_for_message, message["content"], "👎")
                                    st.rerun() # Rerun to update display
                            st.markdown("</div>", unsafe_allow_html=True) # End container div
                # --- End Feedback Section ---

    # --- Handle Chat Input---
    if prompt := st.chat_input("Ask your question here...", disabled=CHAT_DISABLED):
        # Add user message to history with a unique ID
        user_msg_id = f"user_{st.session_state.message_id_counter}"
        st.session_state.message_id_counter += 1
        st.session_state.messages.append({"role": "user", "content": prompt, "id": user_msg_id})

        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_msg_id = st.session_state.messages[-1]["id"]
        assistant_response_exists = any(msg.get("id", "").startswith("asst_") and msg.get("id", "").endswith(user_msg_id.split('_')[-1]) for msg in st.session_state.messages)

        if not assistant_response_exists:
            user_prompt = st.session_state.messages[-1]["content"]

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                sidebar_status_placeholder.markdown(
                    '<script>document.getElementById("sidebar-logo-img").classList.add("pulsing");</script>',
                    unsafe_allow_html=True)

                history_to_send = []
                user_turns = 0
                assistant_turns = 0
                for msg in reversed(st.session_state.messages[:-1]):
                    is_init_msg = msg.get("id") in ["init_0", "init_error", "init_ready", "init_warn"]
                    if msg['role'] == 'user' and user_turns < MAX_CONVERSATION_HISTORY:
                        history_to_send.append(msg)
                        user_turns += 1
                    elif msg['role'] == 'assistant' and assistant_turns < MAX_CONVERSATION_HISTORY:
                        if not is_init_msg: 
                            history_to_send.append(msg)
                            assistant_turns += 1
                    if user_turns >= MAX_CONVERSATION_HISTORY and assistant_turns >= MAX_CONVERSATION_HISTORY:
                        break
                history_to_send.reverse()

                response, sql_script = process_chat_query(user_prompt, history_to_send)


                sidebar_status_placeholder.markdown(
                    '<script>document.getElementById("sidebar-logo-img").classList.remove("pulsing");</script>',
                    unsafe_allow_html=True)

            assistant_msg_id = f"asst_{user_msg_id.split('_')[-1]}" 
            st.session_state.message_id_counter += 1 
            st.session_state.messages.append({"role": "assistant", "content": response, "id": assistant_msg_id, "sql": sql_script or ""  })
            st.rerun()

if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Hello!"}]
    main()
