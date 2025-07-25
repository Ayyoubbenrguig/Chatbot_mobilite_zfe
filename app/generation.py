# generation.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re, logging
from clients import generative_model, CLIENTS_INITIALIZED, CLIENT_INIT_ERROR
from config import SQL_GENERATION_CONFIG, ANSWER_GENERATION_CONFIG, SAFETY_SETTINGS, PROJECT_ID, DATASET_ID, MAX_RESULTS_FOR_CONTEXT



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