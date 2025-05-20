import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import openai
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnableLambda
from pydantic.v1 import BaseModel as LangchainBaseModel, Field as LangchainField
import uuid


# --- Configuration and Initialization ---
LANGSMITH_TRACING_ENABLED = False # Global flag

def load_environment_variables():
    """Loads environment variables from .env file and optionally sets up LangSmith vars."""
    global LANGSMITH_TRACING_ENABLED
    load_dotenv()

    # Core required environment variables
    app_openai_api_key = os.getenv("OPENAI_API_KEY")
    app_openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    app_pinecone_api_key = os.getenv("PINECONE_API_KEY")
    app_pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    core_missing_vars = []
    if not app_openai_api_key: core_missing_vars.append("OPENAI_API_KEY")
    if not app_openai_embedding_model: core_missing_vars.append("OPENAI_EMBEDDING_MODEL")
    if not app_pinecone_api_key: core_missing_vars.append("PINECONE_API_KEY")
    if not app_pinecone_index_name: core_missing_vars.append("PINECONE_INDEX_NAME")

    if core_missing_vars:
        raise ValueError(f"Missing core environment variables: {', '.join(core_missing_vars)}. These are required to run the application. Please ensure they are set in your .env file or deployment secrets.")

    # Optional LangSmith environment variables
    app_langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    app_langchain_project = os.getenv("LANGCHAIN_PROJECT")

    if app_langchain_api_key and app_langchain_project:
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ["LANGCHAIN_API_KEY"] = app_langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = app_langchain_project
        LANGSMITH_TRACING_ENABLED = True
        print("INFO: LangSmith API Key and Project found. Tracing enabled.")
    else:
        LANGSMITH_TRACING_ENABLED = False
        # Unset them if they were somehow set previously without full config, to be safe
        if "LANGCHAIN_TRACING_V2" in os.environ: del os.environ["LANGCHAIN_TRACING_V2"]
        if "LANGCHAIN_API_KEY" in os.environ: del os.environ["LANGCHAIN_API_KEY"]
        if "LANGCHAIN_PROJECT" in os.environ: del os.environ["LANGCHAIN_PROJECT"]
        if "LANGCHAIN_ENDPOINT" in os.environ: del os.environ["LANGCHAIN_ENDPOINT"] # Could interfere if only partially set
        print("INFO: LangSmith API Key or Project not found. Tracing is disabled.")

load_environment_variables() # Load variables on import

# Initialize API clients and models (globally for the module)
# These will use the loaded environment variables.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)
embeddings_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
llm_answer_generator = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
llm_validator = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

set_llm_cache(InMemoryCache()) # Optional: In-memory cache for LLM responses

# --- Pydantic Models for Data Structures ---

class Chunk(BaseModel):
    """Represents a retrieved chunk of text with its metadata."""
    id: str
    text: str
    score: float
    title: str
    source: str

class AnswerValidation(LangchainBaseModel):
    """Pydantic model for parsing LLM validation response (using Pydantic v1 for LangChain compatibility)."""
    is_valid: bool = LangchainField(description="Whether the answer stays within the provided context. True or False.")

class ChunkOutput(BaseModel):
    """Schema for individual chunk representation in the final output."""
    title: str
    text: str
    score: float

class FinalOutput(BaseModel):
    """Schema for the final structured output of the RAG chain."""
    answer: str
    chunks_used: List[ChunkOutput] = Field(default_factory=list)
    validated: Optional[bool] = None
    manual_review_required: Optional[bool] = None


# --- Core Logic Functions ---

def retrieve_and_filter_chunks(user_query: str, top_k: int = 3, score_threshold: float = 0.5) -> List[Chunk]:
    """
    Queries Pinecone for relevant chunks based on the user query and filters them by similarity score.
    Metadata for each chunk is expected to include 'text', 'section_title', and 'source'.
    """
    query_embedding = embeddings_model.embed_query(user_query)
    query_results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    valid_chunks = []
    if query_results.matches:
        for match in query_results.matches:
            if match.score and match.score >= score_threshold: # Ensure score is not None
                metadata = match.metadata if match.metadata else {}
                chunk_text = metadata.get("text", "")
                section_title = metadata.get("section_title", "Unknown Title")
                source_file = metadata.get("source", "Unknown Source")
                
                if not chunk_text:
                    # This print is useful for data quality checks; could be a log entry in a larger system.
                    print(f"Warning: Chunk ID {match.id} has no 'text' in metadata. Skipping.")
                    continue
                valid_chunks.append(Chunk(
                    id=match.id,
                    text=chunk_text,
                    score=match.score,
                    title=section_title,
                    source=source_file
                ))
    return valid_chunks

def format_chunks_for_prompt(chunks: List[Chunk]) -> str:
    """Formats the valid chunks into a string for insertion into the LLM prompt."""
    if not chunks:
        return "No reference documents provided."
    formatted_chunks_list = [f"[{i+1}] Title: {chunk.title}\n{chunk.text}" for i, chunk in enumerate(chunks)]
    return "\n\n".join(formatted_chunks_list)

# --- Prompt Templates ---
LLM_PROMPT_TEMPLATE_STR = """You are a knowledgeable and professional support assistant for account managers at a digital marketing agency.
Your role is to answer the following question using **only** the information from the reference documents provided. Do **not** use outside knowledge, speculate, or guess.

QUESTION:
{user_query}

Your answer must be:
- Accurate and grounded strictly in the documents below
- Concise, clear, and client-ready in tone
- Fully self-contained (do not reference "document 1" or "above")

You are to return me the answer to the question and nothing else.

If the information to answer is not present, respond exactly with:
**"I am not able to answer this question based on the context provided."**

REFERENCE DOCUMENTS:
=====================
{context_str}
====================="""
prompt_template = ChatPromptTemplate.from_template(LLM_PROMPT_TEMPLATE_STR)

VALIDATION_PROMPT_TEMPLATE_STR = """You are reviewing whether an AI-generated answer strictly follows the given context.

QUESTION:
{user_query}

CONTEXT:
{joined_context_from_chunks}

ANSWER:
{llm_answer}

Does the answer stay within the information provided in the context? Respond only with "True" or "False".
Your response should be a JSON object with a single boolean field "is_valid". Example: {{"is_valid": true}}"""
validation_prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT_TEMPLATE_STR)

# --- RAG Chain Definition ---
def create_rag_chain():
    """
    Creates the main RAG (Retrieval Augmented Generation) chain using LangChain Expression Language (LCEL).
    The chain performs the following steps:
    1. Prepares input: Retrieves and filters document chunks from Pinecone based on the user query.
    2. Generates answer: If relevant chunks are found, an LLM generates an answer based on these chunks.
                       If no valid chunks, a fallback message is used.
    3. Validates answer: Another LLM call validates if the generated answer is grounded in the provided context.
    4. Formats output: The final response is structured according to the `FinalOutput` Pydantic model.
    LangSmith tracing is active if LANGCHAIN_API_KEY and LANGCHAIN_PROJECT are set in the environment.
    """

    def prepare_rag_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieves and filters chunks, then formats them for the prompt."""
        user_query = input_data["user_query"]
        retrieved_chunks = retrieve_and_filter_chunks(user_query)
        
        if not retrieved_chunks:
            return { # Early exit if no valid chunks are found
                "user_query": user_query,
                "context_str": "No valid chunks found.", # Placeholder for logging
                "llm_answer_raw": "We do not have enough data to confidently answer this question.",
                "retrieved_chunks": [],
                "skip_llm_and_validation": True
            }
        context_str = format_chunks_for_prompt(retrieved_chunks)
        return {
            "user_query": user_query,
            "context_str": context_str,
            "retrieved_chunks": retrieved_chunks,
            "skip_llm_and_validation": False
        }

    def generate_answer_if_chunks_exist(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates an answer using the LLM if chunks exist; otherwise, passes through pre-set answer."""
        if input_data.get("skip_llm_and_validation"):
            return input_data # Pass through llm_answer_raw and other fields

        answer_generation_chain = prompt_template | llm_answer_generator | StrOutputParser()
        llm_answer_raw = answer_generation_chain.invoke({
            "user_query": input_data["user_query"],
            "context_str": input_data["context_str"]
        })
        return {**input_data, "llm_answer_raw": llm_answer_raw}

    def validate_llm_answer(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates the LLM's answer, skipping if no chunks were used or if it's a predefined fallback."""
        if input_data.get("skip_llm_and_validation"):
            return {**input_data, "validation_result_obj": None, "is_valid_bool": None}

        if input_data["llm_answer_raw"] == "I am not able to answer this question based on the context provided.":
             return {**input_data, "validation_result_obj": None, "is_valid_bool": True} # Instruction-following is valid

        validation_chain = validation_prompt | llm_validator.with_structured_output(AnswerValidation)
        validation_result_obj: AnswerValidation = validation_chain.invoke({
            "user_query": input_data["user_query"],
            "joined_context_from_chunks": input_data["context_str"],
            "llm_answer": input_data["llm_answer_raw"]
        })
        return {**input_data, "validation_result_obj": validation_result_obj, "is_valid_bool": validation_result_obj.is_valid}

    def format_final_output(input_data: Dict[str, Any]) -> FinalOutput:
        """Formats the data into the `FinalOutput` Pydantic model."""
        llm_answer = input_data["llm_answer_raw"]
        retrieved_chunks = input_data["retrieved_chunks"]
        is_valid = input_data.get("is_valid_bool")

        chunks_used_output = [
            ChunkOutput(title=chunk.title, text=chunk.text, score=chunk.score)
            for chunk in retrieved_chunks
        ]
        
        manual_review_required = False
        final_answer_text = llm_answer

        if input_data.get("skip_llm_and_validation"): # No chunks case
            return FinalOutput(
                answer="We do not have enough data to confidently answer this question.",
                chunks_used=[],
                validated=None,
                manual_review_required=None
            )

        if is_valid is False:
            final_answer_text += "\n\n⚠️ This answer may reference information outside of your internal documentation. Manual review is recommended."
            manual_review_required = True
        
        return FinalOutput(
            answer=final_answer_text,
            chunks_used=chunks_used_output,
            validated=is_valid,
            manual_review_required=manual_review_required
        )

    # Define the RAG chain using LCEL
    rag_lcel_chain = (
        RunnableLambda(prepare_rag_input).with_config({"run_name": "PrepareRAGInput"})
        | RunnableLambda(generate_answer_if_chunks_exist).with_config({"run_name": "GenerateAnswer"})
        | RunnableLambda(validate_llm_answer).with_config({"run_name": "ValidateAnswer"})
        | RunnableLambda(format_final_output).with_config({"run_name": "FormatFinalOutput"})
    )
    return rag_lcel_chain


# --- Main Execution Block (for direct script execution and demonstration) ---
if __name__ == "__main__":
    # The LANGSMITH_TRACING_ENABLED flag is set by load_environment_variables()
    if LANGSMITH_TRACING_ENABLED:
        print("INFO: LangSmith tracing is active for this direct execution.")
    else:
        print("INFO: LangSmith tracing is NOT active for this direct execution (API key/project missing).")
    
    # Example query for demonstration
    test_query = "What are the best practices for onboarding new clients?"
    print(f"\nDemonstrating RAG chain with query: \"{test_query}\"\n")

    # Create and invoke the RAG chain
    rag_chain_instance = create_rag_chain()
    
    run_config = None # Default to no specific config if tracing is off
    if LANGSMITH_TRACING_ENABLED:
        trace_id = uuid.uuid4()
        run_config = {
            "metadata": {"user_id": "example_user_main_script", "session_id": str(trace_id)},
            "tags": ["direct_test_run", "example_usage"],
            "run_name": f"RAG_Direct_Query_Run_{trace_id}"
        }
    
    final_result_object: FinalOutput = rag_chain_instance.invoke(
        {"user_query": test_query},
        config=run_config # Pass None if tracing is disabled, LangChain handles it
    )

    print("--- Final Output (Structured Object) ---")
    print(final_result_object.model_dump_json(indent=2))

    # To explore further, you can inspect individual fields:
    # print(f"\nAnswer: {final_result_object.answer}")
    # print(f"Validated: {final_result_object.validated}")
    # print("Chunks Used:")
    # for ch_out in final_result_object.chunks_used:
    #     print(f"  - Title: {ch_out.title}, Score: {ch_out.score:.4f}")

    # Example test cases (commented out by default)
    # print("\n\n--- Example: No Valid Chunks ---")
    # no_chunk_query = "asdasdasdflkjaslkdfjalksdjflkasdjfldkfjgibberishquery"
    # no_chunk_result = rag_chain_instance.invoke({"user_query": no_chunk_query}, config=run_config)
    # print(no_chunk_result.model_dump_json(indent=2))

    # print("\n\n--- Example: Answer Not In Context (but chunks found) ---")
    # query_not_in_docs = "What is the color of the sky on Mars?" # Assuming docs are about digital marketing
    # not_in_docs_result = rag_chain_instance.invoke({"user_query": query_not_in_docs}, config=run_config)
    # print(not_in_docs_result.model_dump_json(indent=2)) 