# 🚗 Geo-Mobility Explorer

A conversational AI chatbot built with Streamlit and Google Cloud (BigQuery + Vertex AI) that answers questions on French geographic data (communes, départements, régions, ZFE zones, fusions/scissions, etc.) and vehicle fleet composition (passenger & light-utility vehicles from 2011 to 2022).

![Interface user](./interface.jpg)

---

## 🎯 Project Overview

- **Data Scope**  
  - **Geography**: All French communes, départements, régions, bassins de vie, arrondissements, ZFE zones, plus historical commune fusions & scissions through 2024.  
  - **Mobility Parc**: National & commune-level stock of passenger (VP) and light-utility (VUL) vehicles by age, fuel type, vignette category, economic sector (2011–2022).  
- **Architecture**  
  1. **Metadata Ingestion** → BigQuery tables  
  2. **Embedding** (Vertex AI) → vector index (FAISS)  
  3. **RAG**: user query → embedding lookup → relevant tables  
  4. **SQL Generation** (Gemini) → BigQuery query → return results  
  5. **Answer Synthesis** (Gemini) → natural-language response  

---

## 🏗️ System Architecture

![Chatbot architecture](./architecture_RAG.jpg)

### 1. Indexing & Retrieval



1. Describe tables in BigQuery  
2. Embed metadata with Vertex AI  
3. Build FAISS index  
4. On user query, retrieve top-k table embeddings  
5. If below relevance threshold → “Unable to answer”  

### 2. Query Generation & Answering



1. Chat history + current query → reformulation LLM  
2. Embed reformulated question → retrieve context  
3. Prompt LLM to generate SQL  
4. Execute on BigQuery → fetch DataFrame  
5. Prompt LLM to produce final natural-language answer  

---

## 🎥 Demo

[![Watch the demo video]](https://youtu.be/KhjDEM5O5rI)  
*Click to watch a short walkthrough of the chatbot in action.*

---

## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone git@github.com:Ayyoubbenrguig/Chatbot_mobilite_zfe.git
  
