# clients.py

import logging, sys
import google.auth
from google.cloud import bigquery, aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
from config import PROJECT_ID, LOCATION, EMBEDDING_MODEL_NAME, MODEL_ID

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
