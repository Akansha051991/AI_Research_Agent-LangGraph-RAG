### 📓 Developer Log: AI Knowledge RAG Agent

This log tracks the technical evolution, hurdles, and architectural decisions made during the development of this project.

## 🛠️ Phase 1: Challenges 
  - **State Synchronization:**
    - **Challenge:** The UI Sidebar failed to update when the database was cleared.
        - **Solution:** Implemented dynamic SQLite polling within the Streamlit sidebar loop to ensure the UI stays in sync with the physical database.

  - **Tool Calling Discipline:**
      - **Challenge:** Smaller models (like Llama 8B) would "over-engineer" tool arguments, leading to frequent API errors.
        - **Solution:** Switched to Llama 3.3 70B and implemented Pydantic Schemas to enforce strict "contracts" between the LLM and the tools.

  - **RAG Priority (The "Wikipedia Trap"):**
      - **Challenge:** The AI tended to prefer its internal knowledge or Wikipedia over the uploaded PDF for well-known topics.
        - **Solution:** Re-engineered the System Prompt to define the Knowledge Base as the "Primary Source of Truth," effectively reducing hallucination and general knowledge bias.

  - **LLM message pattern changed: The new backend often produced:**
        -- **Challenge:** **AIMessage with tool calls → ToolMessages with real info → generic AIMessage (“I’m ready to assist…”).**
          - **Original UI assumed that the last AIMessage is always the real answer, which was no longer true.

##  Phase 2: Lessons Learned


Isolate Debugging: When behavior changes, freeze one side (backend or frontend) while debugging the other. Changing both simultaneously makes it impossible to pinpoint the source of a break.

Define "The Answer" Contract: Don't assume the last AIMessage is the final answer. Either prompt the model to provide a final summary after tool usage or explicitly build the reply from ToolMessages following the last HumanMessage
.
Minimalist System Prompts: Start with a simple, task-focused prompt. Over-constraining with "no outside knowledge" or hard-coded apologies can accidentally override desired behaviors.

Inspect the Message Sequence: The turning point in development was logging and inspecting the full sequence (Human → AI tool_call → Tool → AI). This is the fastest way to understand the model-tool interaction.

Core Loop First: Avoid premature UI optimization. Performance metrics and complex state handling should only be added once the core loop (Input → Backend → Answer) is rock solid to prevent hiding bugs.
