# 🧬 AI Knowledge RAG Agent

A production‑style agentic RAG system built with LangGraph, Pinecone, and multi‑LLM fallbacks to help you ask deep questions over your own PDFs and the web.

# 🤖 AI Knowledge RAG

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-rag-ai.streamlit.app/)
![Project Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square&logo=github)
![Deployment](https://img.shields.io/badge/Deployment-Passing-success?style=flat-square&logo=streamlit)

<img width="560" height="290" alt="AI Knowledge RAG Architecture" src="https://github.com/user-attachments/assets/08662858-700d-4c75-9126-6f2d3ec3f006" />

## ✨ Overview

Upload a PDF (research paper, RFC, design doc) and chat with an AI agent that can:

- Use a **Pinecone‑backed RAG pipeline** to ground answers in your document.
- Call tools like **web search, Wikipedia, YouTube, and weather APIs** when needed.
- Persist full **conversation history and knowledge base** per thread using LangGraph checkpoints.
- Fail over between **Groq Llama 3.3 70B → Gemini 2.0 Flash → GPT‑4o mini** for reliability and latency.


## 🚀 Key Features

- **Agentic RAG with LangGraph** – `chat_node` + `ToolNode` + `tools_condition` let the LLM decide when to call tools vs answer directly from context.
- **Per‑thread PDF knowledge bases** – Each chat gets its own Pinecone namespace, so different PDFs and conversations never leak into each other.
- **Multi‑LLM fallback chain** – Groq → Gemini → OpenAI with `.with_fallbacks`, plus metadata showing which model actually answered and how many tokens were used.  
- **Streaming research UI** – Streamlit chat with live typing effect, a status box that shows tool calls, and perf captions (latency, tokens, active model) under each answer.  
- **Persistent memory & history** – LangGraph’s `SqliteSaver` stores state per `thread_id`, and the sidebar reconstructs past sessions with human‑message‑based titles. 
- **Cost awareness** – Per‑thread token totals and estimated cost metrics in the sidebar.


## Rich Tooling Layer

* **rag_tool –** query your uploaded PDF via Pinecone.
* **TavilySearchResults –** web search for up-to-date information.
* **WikipediaQueryRun –** quick encyclopedic lookups.
* **YouTubeSearchTool –** discover relevant videos for a topic.
* **OpenWeatherMapAPIWrapper –** current weather for a given location.

## 🧱 Architecture Overview
 Frontend (Streamlit)
 
  * st.session_state keys:
   * **thread_id –** current conversation identifier.
   * **message_history –** minimal chat log for rendering UI bubbles.
   * **chat_threads –** cached list of all threads (derived from SQLite checkpoints).
   * **ingested_{thread_id} –** boolean flag indicating whether a PDF has been processed for that thread.

## Chat flow
* User submits a question via st.chat_input.
* Message appended to message_history and rendered as a user bubble.
* chatbot.stream(..., stream_mode="messages") used to:
* Display tool call activity inside a st.status box.
* Stream AI tokens into a placeholder with a cursor-like effect.

## Post-stream
* Final response saved with performance metadata.
* UI-level latency as fallback if backend latency is missing.
* st.rerun() to keep state and UI consistent.

## Sidebar
* PDF upload or “Remove PDF” per active thread.
* New Chat (resets thread_id + history).
* Delete History (clears SQLite tables and cached checkpointer).
* Thread history list (buttons that load stored messages from chatbot state and reconstruct message_history).


## ⚙️ Tech Stack

- **Frameworks:** LangGraph, LangChain, Streamlit  
- **LLMs:** Groq Llama‑3.3‑70B, Google Gemini 2.0 Flash, OpenAI GPT‑4o mini  
- **RAG:** Pinecone Vector Store + `multilingual-e5-large` embeddings  
- **Storage:** SQLite (LangGraph checkpoints and thread history)  
- **Tools:** Tavily search, Wikipedia API, YouTube search, OpenWeatherMap

## 🚀 Getting Started

### 1. Clone the repository
```
git clone https://github.com/<your-username>/agentic-rag-ai.git
cd agentic-rag-ai

```

### 2. Create and Activate a Virtual Environment
```
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
```
### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure Environment Variables

*  Create a .env file in the project root:
```
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
# Optional if Tavily / OpenWeather / etc. require keys in your setup
TAVILY_API_KEY=your_tavily_key
OPENWEATHERMAP_API_KEY=your_openweather_key

```
* Make sure the .env is loaded via load_dotenv() (already present in the backend).

### 5. Run the App Locally
```
streamlit run app.py
```
* Then open the URL shown in your terminal (typically http://localhost:8501).

 ### 5. Verify Observability
 
After interacting with the bot, visit your LangSmith dashboard to see the execution traces:

[![LangSmith](https://img.shields.io/badge/LangSmith-Observability-orange?style=flat-square&logo=langchain&logoColor=white)](https://eu.smith.langchain.com/)

###  🧪 How to Use the App
 Start a new chat  ---> Upload a PDF (optional but recommended)  ---> Ask a question --> Inspect responses --> Monitor usage --> Navigate history ---> Reset or clear


 ### 📦 Deployment
 
The project is already deployed on Streamlit Community Cloud:

> **Try it out here:** [AI Knowledge RAG Live Demo](https://agentic-rag-ai.streamlit.app/)

* To deploy your own instance:
  * Push your code to a public GitHub repository.
  * Go to share.streamlit.io.
  * Connect your repo and choose the main file (e.g., app.py).
  * Add your secrets and environment variables in the Streamlit Cloud settings.
  * Deploy and share your URL
 
    
### 🛡️ Notes & Best Practices
 * API costs & rate limits
 * You are calling multiple providers (Groq, Google, OpenAI, Pinecone, Tavily, etc).
 * Keep an eye on quotas; adjust model choices or add caching if needed
 
### Visual Representation 
 
<img width="6000" height="5000" alt="Chat Workflow Integration-2026-03-04-010920" src="https://github.com/user-attachments/assets/5b2edc8c-1cd4-4690-819c-810fe02f1e12" />

## 🧪 Evaluation & Metrics

- **Runtime metrics:** Each assistant message includes latency, token count, and the active model, collected from `usage_metadata`.  
- **Cost tracking:** The sidebar shows aggregated tokens and an estimated cost per thread.  
- **Planned:** A small evaluation script that runs a fixed set of questions against a sample PDF to compare:
  - RAG vs no‑RAG,
  - Groq‑only vs multi‑LLM fallback,
  - Different `k` values in similarity search.
 
  
      <img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/98912efd-f844-4c00-b1e9-c9628416deb0" />


### 🗓️ Upcoming Milestones

- [x] **Scale & Performance:** Add support for multiple-PDF uploads and optimize the pipeline to handle larger files more gracefully.
- [x] **The Evaluation Layer:** Introduce a formal evaluation framework, such as RAGAS, to move from "vibes-based" testing to quantified   metrics.
- [x] **Semantic Chunking:** Measure the impact on context precision when switching from naive character-based splitting.
- [x] **Hybrid Retrieval:** Implement BM25 + Vector Search to improve keyword-based retrieval accuracy.
- [x] **Retrieve → evaluate “is this enough / relevant?” → refine query & re‑retrieve → answer.** 
- [x] “self‑corrective” RAG loop.

  
## 📖 Technical Evolution (Dev Log)
This log tracks the technical evolution, hurdles, and architectural decisions made during the development of this project.



