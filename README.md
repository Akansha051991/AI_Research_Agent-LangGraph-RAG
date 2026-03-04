# 🧬 AI Knowledge RAG – Research Agent with LangGraph & Streamlit
An AI-first research assistant that combines Retrieval-Augmented Generation (RAG), multi-model fallback (Groq, Gemini, OpenAI), and live tools (web search, Wikipedia, YouTube, weather) in a clean Streamlit UI.

## 🌟 Key Features
* **Agentic Workflow:** Agentic RAG with LangGraph

  * **Stateful** agent built on StateGraph with a chat_node and ToolNode wired through tools_condition.
  * **Uses** SqliteSaver as a checkpointer to persist conversations per thread_id

* **Multi-model fallback (Resilient LLM stack)**
  * **Primary:** Groq llama-3.3-70b-versatile (fast, strong reasoning).
  * **Secondary:** Google gemini-2.0-flash.
  * **Tertiary:** OpenAI gpt-4o-mini.
  
   All three are bound to the same toolset and orchestrated via .with_fallbacks(...)


  * **PDF Knowledge Base (Per-Thread RAG)**
      Upload a PDF per chat thread and build a Pinecone vector namespace keyed by thread_id.
      Chunking via RecursiveCharacterTextSplitter and embeddings via multilingual-e5-large (Pinecone embeddings).
      rag_tool always queried first (enforced in the system prompt) for knowledge-base-centric answers.

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



## 🚀 Getting Started

### 1. Clone the repository
```
git clone https://github.com/<your-username>/agentic-rag-ai.git
cd agentic-rag-ai

```

### 2. Create and Activate a Virtual Environment
```
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
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
 
<img width="7138" height="6365" alt="Chat Workflow Integration-2026-03-04-010920" src="https://github.com/user-attachments/assets/5b2edc8c-1cd4-4690-819c-810fe02f1e12" />


