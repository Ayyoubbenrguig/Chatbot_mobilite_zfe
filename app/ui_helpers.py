# ui_helpers.py

import streamlit as st
import base64
from config import BACKGROUND_IMAGE_PATH, LOGO_PATH
import logging
import faiss
from config import DATASET_ID, TABLE_PREFIX_FILTER
from clients import CLIENTS_INITIALIZED, CLIENT_INIT_ERROR, embedding_model
from bigquery_utils import query_bigquery_metadata
from faiss_index import build_faiss_index

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