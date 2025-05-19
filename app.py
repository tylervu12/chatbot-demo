import gradio as gr
import os
import uuid

# Import necessary components from your main RAG pipeline script
from main import (
    create_rag_chain,
    load_environment_variables,
    FinalOutput, # The output schema of our RAG chain
    ChunkOutput # Schema for chunks used - this is part of FinalOutput
)

# Load environment variables (OpenAI, Pinecone, LangSmith keys)
# This is crucial for the RAG chain to function.
try:
    load_environment_variables()
    print("Environment variables loaded successfully for Gradio app.")
    # Create the RAG chain instance once when the app loads
    rag_chain = create_rag_chain()
    print("RAG chain created successfully.")
    CHAIN_LOADED_SUCCESSFULLY = True
except ValueError as e:
    print(f"ERROR during app initialization: Missing environment variables: {e}")
    CHAIN_LOADED_SUCCESSFULLY = False
    # Define a placeholder rag_chain if loading failed, so the app can still run (with errors displayed)
    def faulty_rag_chain(inputs):
        return FinalOutput(
            answer=f"Error: RAG Chain could not be initialized. {e}",
            chunks_used=[],
            validated=None,
            manual_review_required=None
        )
    rag_chain = faulty_rag_chain
except Exception as e:
    print(f"ERROR during app initialization: An unexpected error occurred: {e}")
    CHAIN_LOADED_SUCCESSFULLY = False
    def faulty_rag_chain_unexpected(inputs):
        return FinalOutput(
            answer=f"Error: RAG Chain faced an unexpected error during initialization. {e}",
            chunks_used=[],
            validated=None,
            manual_review_required=None
        )
    rag_chain = faulty_rag_chain_unexpected

def get_chatbot_response(user_query: str):
    """
    Gets a response from the RAG chain and formats it for Gradio output.
    """
    if not CHAIN_LOADED_SUCCESSFULLY:
        # If the chain didn't load, rag_chain is a placeholder that returns an error message.
        # The placeholder function expects a dict, so we provide one.
        output: FinalOutput = rag_chain({"user_query": user_query}) 
        return output.answer, "Please check server logs for initialization errors.", ""

    if not user_query.strip():
        return "Please enter a question.", "", ""

    print(f"Received query: {user_query}")
    
    # Generate a unique trace ID for this specific run if desired for LangSmith
    trace_id = uuid.uuid4()
    config = {
        "metadata": {"user_id": "gradio_user", "session_id": str(trace_id)}, 
        "tags": ["gradio_interaction"], 
        "run_name": f"RAG_Gradio_Query_{trace_id}"
    }

    try:
        result: FinalOutput = rag_chain.invoke({"user_query": user_query}, config=config)
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed server-side logging
        return f"An error occurred while processing your question: {e}", "", ""

    answer_text = result.answer
    sources_text = "### Sources Used:\n\n"
    if result.chunks_used:
        for i, chunk in enumerate(result.chunks_used):
            sources_text += f"**Source {i+1}: {chunk.title}** (Score: {chunk.score:.4f})\n"
            sources_text += f"> {chunk.text[:300]}...\n\n" # Displaying first 300 chars of chunk text
    else:
        sources_text += "No specific document chunks were heavily relied upon for this answer, or the answer is a fallback response."
    
    validation_info = ""
    if result.validated is False:
        validation_info = "⚠️ This answer may not be fully aligned with provided documents. Manual review recommended."
    elif result.validated is True:
        validation_info = "✅ Answer validated against provided documents."
    else: # validated is None (e.g. no chunks path)
        validation_info = ""

    return answer_text, sources_text, validation_info

# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 RAG Chatbot Demo
    Ask a question to get an answer based on our knowledge base.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(label="Your Question", placeholder="e.g., What are the best practices for SEO?")
            submit_button = gr.Button("Ask Question", variant="primary")
        with gr.Column(scale=3):
            gr.Markdown("### Answer")
            answer_output = gr.Markdown(value="Your answer will appear here...")
            validation_output = gr.Markdown()

    gr.Markdown("---_Sources_---")
    sources_output = gr.Markdown()
    
    # Linking the button to the function
    submit_button.click(
        fn=get_chatbot_response,
        inputs=question_box,
        outputs=[answer_output, sources_output, validation_output]
    )
    
    gr.Examples(
        examples=[
            "What are the best practices for onboarding new clients?",
            "How do you handle ad disapprovals?",
            "What is the process for monthly reporting?",
            "Tell me about our agency's approach to keyword research."
        ],
        inputs=question_box
    )

if __name__ == "__main__":
    print("Launching Gradio interface...")
    # For local testing. When deploying to Hugging Face, HF Spaces runs app.py directly.
    demo.launch(share=False) # Set share=True to get a public link for temporary sharing 