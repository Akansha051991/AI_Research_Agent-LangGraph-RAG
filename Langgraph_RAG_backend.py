import os
import sqlite3
import streamlit as st
from typing import TypedDict, Annotated, Optional
import time
import logging
from dotenv import load_dotenv

# Core LangChain & LangGraph
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, trim_messages,HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

# RAG / PDF Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

# LLM & Tools
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool
from langchain_community.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. STATE DEFINITION (Fixed) ---
class ChatState(TypedDict):
    # messages stores history; add_messages ensures new messages append to the list
    messages: Annotated[list[BaseMessage], add_messages]
    # metadata stores performance data; Optional allows it to be empty initially
    metadata: Optional[dict] 

#1. Primary Model: Groq (Llama 3.3)
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=2,
    timeout=20
)
#2. Tertiary Model: OpenAI (GPT-4o-mini)
openai_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_retries=3,
    timeout=60
)
# 3. Secondary Model: Google Gemini (2.0 Flash)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_retries=2,
    timeout=30
)


embeddings = PineconeEmbeddings(
    model="multilingual-e5-large", 
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "rag-langgraph"


def ingest_pdf(file_path: str, thread_id: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=index_name,
        namespace=str(thread_id)
    )
    return {"status": "success", "chunks": len(splits)}
# --- 3. TOOLS ---
# --- GLOBAL INITIALIZATION (Do this once outside the tool) ---
# Initialize the Pinecone VectorStore globally so it's ready to go
vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings
)

@tool
def rag_tool(query: str, config: RunnableConfig):
    """Search the PDF knowledge base for technical or document-specific information."""
    t_id = config.get("configurable", {}).get("thread_id")
    # Use the GLOBAL vectorstore and just specify the namespace during the search
    docs = vectorstore.similarity_search(
        query, 
        k=4, 
        namespace=str(t_id) # Namespace keeps data isolated per thread
    )
    formatted_docs = [f"[Source: Page {d.metadata.get('page', 0) + 1}]\nContent: {d.page_content}" for d in docs]
    return "\n\n---\n\n".join(formatted_docs) if formatted_docs else "No matching content found."

@tool
def youtube_search(query: str):
    """Search YouTube. Input: search query string. Returns video URLs."""
    yt = YouTubeSearchTool()
    return yt.run(f"{query}, 2")

@tool
def get_weather(location: str):
    """Get current weather for a location. Input: 'City, Country' string."""
    try:
        weather_api = OpenWeatherMapAPIWrapper()
        raw_response = weather_api.run(location)
        if "Temperature" in raw_response:
            return f"### 🌡️ Weather for {location}\n{raw_response}"
        return raw_response
    except Exception as e:
        # TEMP: return the real error so we can debug deployment
        return f"Unable to retrieve weather data for {location}."


search_tool = TavilySearchResults(max_results=2)
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)


# Define your tools list 
tools = [search_tool, wiki_tool, get_weather, youtube_search, rag_tool] 


# Bind tools to EACH model
groq_with_tools = groq_llm.bind_tools(tools)
gemini_with_tools = gemini_llm.bind_tools(tools)
openai_with_tools = openai_llm.bind_tools(tools)


# Create the resilient chain
# It tries Groq -> then Gemini -> then OpenAI
llm_with_tools = groq_with_tools.with_fallbacks([openai_with_tools, gemini_with_tools])


# --- 4. GRAPH NODES ---
def chat_node(state: ChatState, config: RunnableConfig): 
    logger.info(f"--- Calling LLM for Thread: {config['configurable']['thread_id']} ---")
    start_time = time.time()
    
    # Trim history to manage context window
    trimmed_messages = trim_messages(
        state["messages"],
        max_tokens=2000,
        strategy="last",
        token_counter=lambda msgs: sum(len(m.content) for m in msgs) // 4, 
        include_system=True,
        start_on="human" # <--- CRITICAL for API stability
    )
    
    system_instruction = SystemMessage(content=(
    "You are a Research Assistant. ALWAYS check the 'rag_tool' tool first "
    "for any user query. Only use Wikipedia if the knowledge_base tool returns "
    "no relevant information. If the user mentions 'knowledge base', DO NOT "
    "use Wikipedia."
))

    messages_with_instruction = [system_instruction] + trimmed_messages
    
    try:
        # Use the combined llm_with_tools we created above
        response = llm_with_tools.invoke(messages_with_instruction, config=config)
        
        # Safely extract usage metadata (different models use different keys)
        usage = getattr(response, "usage_metadata", {})
        total_tokens = usage.get("total_tokens", 0)
        generation_latency = time.time() - start_time
        
        return {
            "messages": [response],
            "metadata": {
                "generation_latency": f"{generation_latency:.2f}s",
                "tokens": total_tokens,
                "active_model": response.response_metadata.get("model_name", "Unknown Model") # Tracks which model actually answered
            
            }
        }
    except Exception as e:
        logger.error(f"Chain Failure: {e}")
        return {"messages": [AIMessage(content="I'm having trouble connecting to my brains. Please try again.")]}
        

# --- 5. GRAPH CONSTRUCTION ---
@st.cache_resource
def get_checkpointer():
    conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
    return SqliteSaver(conn=conn)

checkpointer = get_checkpointer()

workflow = StateGraph(ChatState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chat_node")
workflow.add_conditional_edges("chat_node", tools_condition)
workflow.add_edge("tools", "chat_node")

chatbot = workflow.compile(checkpointer=checkpointer)

# --- 6. UTILITIES ---
def retrieve_all_threads():
    conn = sqlite3.connect(database="chatbot.db")
    cursor = conn.cursor()
    thread_data = [] # Change to list of dicts
    
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        
        for tid in threads:
            # Get the state of this specific thread
            state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
            msgs = state.values.get("messages", [])
            
            # Find the first human message to use as the title
            title = "New Chat" # Default
            for m in msgs:
                if isinstance(m, HumanMessage) and m.content:
                    # Truncate to 20 chars so it fits in the sidebar
                    title = (m.content[:20] + "...") if len(m.content) > 20 else m.content
                    break
            
            thread_data.append({"id": tid, "title": title})
            
        return thread_data
    except Exception as e:
        logger.error(f"Error retrieving threads: {e}")
        return []
    finally:
        conn.close()


def clear_all_history():
    try:
        conn = sqlite3.connect(database="chatbot.db", timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            if "checkpoint" in table or "writes" in table:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"Cleared table: {table}")
        
        conn.commit()
        conn.close()

        get_checkpointer.clear() 
        return True
    except Exception as e:
        st.error(f"Database Error: {e}")
        return False
    
