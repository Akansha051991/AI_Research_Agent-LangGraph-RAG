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
