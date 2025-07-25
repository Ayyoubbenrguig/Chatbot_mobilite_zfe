# app.py

import streamlit as st
from config import APP_TITLE, LOGO_PATH, FAQS
from ui_helpers import set_page_background_and_styles, get_base64_of_bin_file
from clients import CLIENTS_INITIALIZED, CLIENT_INIT_ERROR
from bigquery_utils import query_bigquery_metadata
from faiss_index import build_faiss_index
from chat_logic import process_chat_query, initialize_chatbot_engine
from feedback import log_feedback
from config import BACKGROUND_IMAGE_PATH, MODEL_ID, TABLE_PREFIX_FILTER, MAX_CONVERSATION_HISTORY, BQ_DATASET_REF


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
        - **Scope:** Mobility patterns across France  
        - **Surveys:** 2008 & 2019 household travel data  
        - **Tables indexed:** France_Table_*\  
        (e.g. communes, départements, ZFE zones, vehicle parc, etc.)  
        - **Rows:** ~3.5 M records total  
        """)
        
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

        # Model & Config
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
                                    log_feedback(msg_id, current_user_query, message["content"], "👍")
                                    st.rerun() # Rerun to update display
                            with col_down:
                                if st.button("👎", key=f"{feedback_key_base}_down", help="Mark as not helpful"):
                                    st.session_state.feedback_given[msg_id] = "👎"
                                    log_feedback(msg_id, current_user_query, message["content"], "👎")
                                    st.rerun() # Rerun to update display
                            st.markdown("</div>", unsafe_allow_html=True) # End container div
                # --- End Feedback Section ---


    # --- Display FAQs---
    with st.expander("💡 Frequently Asked Questions (Examples)"):
        faq_html = "<ul>"
        for faq in FAQS:
            faq_html += f"<li>{faq}</li>"
        faq_html += "</ul>"
        st.markdown(faq_html, unsafe_allow_html=True)


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
            st.session_state.messages.append({"role": "assistant", "content": response, "id": assistant_msg_id})
            st.rerun()

if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"assistant","content":"Hello!"}]
    main()
