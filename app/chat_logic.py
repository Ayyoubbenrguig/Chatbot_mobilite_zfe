# chat_logic.py
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import streamlit as st
import logging
import faiss
from config import DATASET_ID, TABLE_PREFIX_FILTER
from clients import CLIENTS_INITIALIZED, CLIENT_INIT_ERROR, embedding_model
from bigquery_utils import query_bigquery_metadata
from faiss_index import check_relevance_faiss, find_relevant_tables_faiss, build_faiss_index
from generation import generate_sql_query_gemini, generate_contextual_answer_gemini
from bigquery_utils import execute_sql_query



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
