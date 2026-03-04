### 📓 Developer Log: AI Knowledge RAG Agent

The backend is built with LangGraph (StateGraph + ToolNode + SqliteSaver) and Pinecone; these notes explain the problems hit while wiring that architecture into a streaming Streamlit UI.

## 🛠️ Phase 1: Challenges 

- **State synchronization**
  - **Challenge:** The UI sidebar failed to update when the SQLite database was cleared.
  - **Solution:** Implemented dynamic SQLite polling inside the Streamlit sidebar loop so the UI always reflects the real database state.

- **Tool calling discipline**
  - **Challenge:** Smaller models (like Llama 8B) would over‑engineer tool arguments, causing frequent API errors.
  - **Solution:** Switched to Llama 3.3 70B and introduced Pydantic‑style schemas to enforce strict contracts between the LLM and tools.

- **RAG priority (“Wikipedia trap”)**
  - **Challenge:** The AI preferred its own knowledge or Wikipedia over the uploaded PDF for well‑known topics.
  - **Solution:** Re‑engineered the system prompt so the PDF knowledge base is the primary source of truth, reducing hallucinations and general‑knowledge bias.

- **LLM message pattern changes**
  - **Challenge:** The backend began producing sequences like:  
    `AIMessage (tool calls) → ToolMessages (real info) → generic AIMessage (“I’m ready to assist…”)`,  
    while the UI assumed “the last AIMessage is always the real answer”.
  - **Solution:** Updated the UI logic and internal “answer contract” so the final response is built from the correct AIMessage / ToolMessages after the last HumanMessage.

## 🧠 Phase 2: Lessons Learned

- **Isolate debugging:** When behavior changes, freeze either the backend or frontend and debug one side at a time. Changing both hides the real bug.  
- **Define the “answer contract”:** Never assume the last AIMessage is the final answer. Either prompt the model to emit a clear final summary after tools, or explicitly construct the reply from the ToolMessages that follow the last HumanMessage. 
- **Minimalist system prompts:** Start with a simple, task‑focused system prompt. Over‑constraining with “no outside knowledge” or hard‑coded apologies can override desired tool‑using behavior.
- **Inspect the message sequence:** Logging the full Human → AI (tool_call) → Tool → AI sequence was the turning point for understanding model–tool interactions and fixing subtle bugs. 
- **Core loop first:** Make the core loop (Input → Backend → Answer) rock solid before adding complex UI features or performance metrics, otherwise you risk decorating a broken workflow.
