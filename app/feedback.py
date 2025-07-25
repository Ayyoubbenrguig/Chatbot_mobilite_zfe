# feedback.py

from datetime import datetime
from datetime import datetime
import datetime
import os
import logging
import streamlit as st

from google.cloud import bigquery

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


