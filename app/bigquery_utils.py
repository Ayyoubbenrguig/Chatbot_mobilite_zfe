# bigquery_utils.py
from typing import Tuple, Dict, Optional
import pandas as pd
import logging
from pandas.api.types import is_numeric_dtype
from google.cloud import bigquery
from config import PROJECT_ID, TABLE_PREFIX_FILTER, MAX_SAMPLE_VALUES
from clients import bq_client, CLIENTS_INITIALIZED, CLIENT_INIT_ERROR



# Step 1: Metadata Fetching
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


# Step 6: Execute SQL Query:

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
