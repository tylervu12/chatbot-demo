import gradio as gr
import os
import uuid

# Import necessary components from your main RAG pipeline script
from main import (
    create_rag_chain,
    load_environment_variables,
    FinalOutput, # The output schema of our RAG chain
    LANGSMITH_TRACING_ENABLED # Import the flag from main.py
)

# --- App Initialization ---
# Load environment variables and initialize the RAG chain once on startup.
# This is crucial for the RAG chain to function and improves responsiveness.
CHAIN_LOADED_SUCCESSFULLY = False
RAG_CHAIN_INIT_ERROR_MESSAGE = "RAG chain is not initialized. Please check server logs."

try:
    # load_environment_variables() is called when main.py is imported,
    # so LangSmith status is already determined.
    print("INFO: Environment variables (including optional LangSmith) processed via main.py import.")
    if LANGSMITH_TRACING_ENABLED:
        print("INFO: LangSmith tracing is ENABLED for this Gradio app session.")
    else:
        print("INFO: LangSmith tracing is DISABLED for this Gradio app session (API key/project missing).")

    rag_chain_instance = create_rag_chain()
    print("INFO: RAG chain created successfully.")
    CHAIN_LOADED_SUCCESSFULLY = True
except ValueError as e:
    # Specific error for missing environment variables
    RAG_CHAIN_INIT_ERROR_MESSAGE = f"ERROR: RAG Chain initialization failed due to missing environment variables: {e}. Ensure .env file is correct or secrets are set in deployment."
    print(RAG_CHAIN_INIT_ERROR_MESSAGE)
except Exception as e:
    # Catch-all for other unexpected errors during initialization
    RAG_CHAIN_INIT_ERROR_MESSAGE = f"ERROR: RAG Chain faced an unexpected error during initialization: {e}."
    print(RAG_CHAIN_INIT_ERROR_MESSAGE)
    import traceback
    traceback.print_exc() # Print full traceback to server logs for debugging

# Define a placeholder function if chain loading failed, to provide a graceful error in the UI
if not CHAIN_LOADED_SUCCESSFULLY:
    def faulty_rag_chain_placeholder(inputs: dict) -> FinalOutput:
        return FinalOutput(
            answer=RAG_CHAIN_INIT_ERROR_MESSAGE,
            chunks_used=[],
            validated=None,
            manual_review_required=None
        )
    rag_chain_instance = faulty_rag_chain_placeholder

# --- Gradio Interaction Logic ---
def get_chatbot_response(user_query: str):
    """
    Processes the user's query using the RAG chain and formats the response for Gradio display.
    Handles errors gracefully and provides informative messages.
    Optionally sends trace data to LangSmith if configured.
    """
    if not user_query.strip():
        return "Please enter a question.", "", ""

    print(f"INFO: Received query from Gradio interface: \"{user_query}\"")
    
    if not CHAIN_LOADED_SUCCESSFULLY:
        # The RAG chain is already the placeholder function if initialization failed.
        # The placeholder expects a dict, so provide one.
        result: FinalOutput = rag_chain_instance({"user_query": user_query}) 
        return result.answer, "Initialization Error. Check server logs.", ""

    run_config = None
    if LANGSMITH_TRACING_ENABLED:
        trace_id = uuid.uuid4()
        run_config = {
            "metadata": {"user_id": "gradio_user", "session_id": str(trace_id), "interaction_type": "chatbot_query"},
            "tags": ["gradio_interaction", "chatbot"],
            "run_name": f"RAG_Gradio_Query_{trace_id}"
        }

    try:
        result: FinalOutput = rag_chain_instance.invoke({"user_query": user_query}, config=run_config)
    except Exception as e:
        error_message = f"An error occurred while processing your question: {str(e)[:200]}..."
        print(f"ERROR: Error invoking RAG chain: {e}")
        # import traceback # Already imported if init failed
        # traceback.print_exc() # Uncomment for detailed server-side debugging if needed
        return error_message, "Could not retrieve sources due to an error.", "Error processing request."

    # Format the answer
    answer_text = result.answer

    # Format sources
    sources_text = "### Sources Used:\n\n"
    if result.chunks_used:
        for i, chunk in enumerate(result.chunks_used):
            # Ensure chunk object has expected attributes, provide defaults if not (robustness)
            title = getattr(chunk, 'title', 'Unknown Title')
            score = getattr(chunk, 'score', 0.0) # Default score to 0.0 if missing
            text_snippet = getattr(chunk, 'text', 'No text preview available.')[:300] + "..."
            
            sources_text += f"**Source {i+1}: {title}** (Score: {score:.4f})\n"
            sources_text += f"> {text_snippet}\n\n"
    else:
        sources_text += "No specific document chunks were identified as primary sources, or the answer is a general fallback."
    
    # Format validation information
    validation_info = ""
    if result.validated is False:
        validation_info = "⚠️ This answer may reference information outside of the provided documents. Manual review recommended."
    elif result.validated is True:
        validation_info = "✅ Answer validated against provided documents."
    # No specific message if validated is None (e.g., for fallback messages where validation might not apply)

    return answer_text, sources_text, validation_info

# --- Gradio Interface Definition ---
# Using gr.Blocks for a more custom layout.
# Theme can be adjusted (e.g., gr.themes.Default(), gr.themes.Glass(), gr.themes.Monochrome())
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Chatbot") as demo:
    gr.Markdown("""
    # 🤖 RAG Chatbot Demo
    Ask a question about our services or knowledge base. The chatbot will use internal documents to find an answer.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your Question", 
                placeholder="e.g., What are the best practices for SEO?",
                lines=3
            )
            submit_button = gr.Button("Get Answer", variant="primary")
        with gr.Column(scale=3):
            gr.Markdown("### Answer from Knowledge Base")
            answer_output = gr.Markdown(value="Your answer will appear here once you ask a question.")
            validation_output = gr.Markdown() # For validation status

    gr.Markdown("---_Retrieved Context & Sources_---")
    sources_output = gr.Markdown(value="Details about the information used will appear here.")
    
    # Event listener for the button click
    submit_button.click(
        fn=get_chatbot_response,
        inputs=question_box,
        outputs=[answer_output, sources_output, validation_output],
        api_name="ask_chatbot" # Exposes this function via API if interface is launched with share=True or embedded
    )
    
    # Provide example questions for users
    gr.Examples(
        examples=[
            "What are the best practices for onboarding new clients?",
            "How do you handle ad disapprovals?",
            "What is our agency's approach to content marketing strategy?",
            "Tell me about the monthly performance reporting process."
        ],
        inputs=question_box,
        label="Example Questions (click to try)"
    )

# --- Main Execution Block (for local testing) ---
if __name__ == "__main__":
    print("INFO: Launching Gradio interface locally...")
    # To create a public link for temporary sharing, set share=True.
    # For Hugging Face Spaces, HF handles the server; this block is mainly for local dev.
    demo.launch(share=False) 