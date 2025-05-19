import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
import openai
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.tracers import LangChainTracer
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pydantic.v1 import BaseModel as LangchainBaseModel, Field as LangchainField
import uuid


# --- Configuration and Initialization ---
def load_environment_variables():
    """Loads environment variables from .env file and sets up LangSmith vars."""
    load_dotenv()

    # Fetch all potentially required environment variables first
    app_openai_api_key = os.getenv("OPENAI_API_KEY")
    app_openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    app_pinecone_api_key = os.getenv("PINECONE_API_KEY")
    app_pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    app_langchain_api_key = os.getenv("LANGCHAIN_API_KEY") # For LangSmith
    app_langchain_project = os.getenv("LANGCHAIN_PROJECT") # For LangSmith

    # Check for missing critical environment variables
    missing_vars = []
    if not app_openai_api_key: missing_vars.append("OPENAI_API_KEY")
    if not app_openai_embedding_model: missing_vars.append("OPENAI_EMBEDDING_MODEL")
    if not app_pinecone_api_key: missing_vars.append("PINECONE_API_KEY")
    if not app_pinecone_index_name: missing_vars.append("PINECONE_INDEX_NAME")
    if not app_langchain_api_key: missing_vars.append("LANGCHAIN_API_KEY")
    if not app_langchain_project: missing_vars.append("LANGCHAIN_PROJECT")

    if missing_vars:
        raise ValueError(f"Missing critical environment variables: {', '.join(missing_vars)}. Please ensure they are set in your .env file.")

    # Set LangSmith specific environment variables for LangChain to pick up
    # These must be set in os.environ for LangChain's auto-tracing capabilities
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_API_KEY"] = app_langchain_api_key # Safe, as it's checked above
    os.environ["LANGCHAIN_PROJECT"] = app_langchain_project # Safe, as it's checked above

load_environment_variables()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Initialize OpenAI Embeddings
embeddings_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

# Initialize LangSmith Tracer (optional, LangChain handles it if env vars are set)
# tracer = LangChainTracer() # You might not need to instantiate it explicitly

# Initialize LLMs
llm_answer_generator = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
llm_validator = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

# Optional: In-memory cache for LLM responses to reduce redundant calls during development/testing
set_llm_cache(InMemoryCache())

# --- Pydantic Models for Data Structures ---

class Chunk(BaseModel):
    """Represents a retrieved chunk of text."""
    id: str
    text: str
    score: float
    title: str
    source: str

class AnswerValidation(LangchainBaseModel):
    """Pydantic model for parsing LLM validation response."""
    is_valid: bool = LangchainField(description="Whether the answer stays within the provided context. True or False.")

class ChunkOutput(BaseModel):
    """Schema for individual chunk in the final output."""
    title: str
    text: str
    score: float

class FinalOutput(BaseModel):
    """Schema for the final output to the front end."""
    answer: str
    chunks_used: List[ChunkOutput] = Field(default_factory=list)
    validated: Optional[bool] = None
    manual_review_required: Optional[bool] = None


# --- Core Logic Functions ---

# Step 2: Embed User Query (Handled by LangChain PineconeVectorStore)

# Step 3: Query Pinecone
def retrieve_and_filter_chunks(user_query: str, top_k: int = 3, score_threshold: float = 0.5) -> List[Chunk]:
    """
    Queries Pinecone for relevant chunks and filters them by score.
    Pinecone results typically have 'id', 'score', 'values' (embedding), and 'metadata'.
    We expect metadata to contain 'text', 'section_title', and 'source'.
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
            if match.score >= score_threshold:
                metadata = match.metadata if match.metadata else {}
                chunk_text = metadata.get("text", "") # Ensure 'text' key exists
                section_title = metadata.get("section_title", "N/A")
                source_file = metadata.get("source", "N/A")
                
                if not chunk_text:
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

# Step 4: Dynamically Construct the Prompt
def format_chunks_for_prompt(chunks: List[Chunk]) -> str:
    """Formats the valid chunks for insertion into the LLM prompt."""
    if not chunks:
        return "No reference documents provided."
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append(f"[{i+1}] Title: {chunk.title}\n{chunk.text}")
    return "\n\n".join(formatted_chunks)

LLM_PROMPT_TEMPLATE = """You are a knowledgeable and professional support assistant for account managers at a digital marketing agency.
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

prompt_template = ChatPromptTemplate.from_template(LLM_PROMPT_TEMPLATE)

# Step 5: Call LLM to Generate Answer (Handled by LangChain chain)

# Step 6: Validate Answer Using a Second LLM
VALIDATION_PROMPT_TEMPLATE = """You are reviewing whether an AI-generated answer strictly follows the given context.

QUESTION:
{user_query}

CONTEXT:
{joined_context_from_chunks}

ANSWER:
{llm_answer}

Does the answer stay within the information provided in the context? Respond only with "True" or "False".
Your response should be a JSON object with a single boolean field "is_valid". Example: {{"is_valid": true}}"""

validation_prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT_TEMPLATE)

# Step 8: Log to LangSmith (LangChain handles this automatically with environment variables)

# --- Main RAG Chain ---
def create_rag_chain():
    """Creates the main RAG chain using LangChain Expression Language."""

    def prepare_rag_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares the input for the RAG chain.
        Retrieves chunks and formats them for the prompt.
        """
        user_query = input_data["user_query"]
        retrieved_chunks = retrieve_and_filter_chunks(user_query)
        
        if not retrieved_chunks:
            return { # Early exit if no chunks
                "user_query": user_query,
                "context_str": "No valid chunks found.", # Placeholder for LangSmith
                "llm_answer": "I am not able to answer this question based on the context provided.",
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
        """Generates an answer using the LLM only if chunks were found."""
        if input_data.get("skip_llm_and_validation"):
            # llm_answer is already set in prepare_rag_input
            return {**input_data, "llm_answer_raw": input_data["llm_answer"]}

        # If not skipping, proceed with LLM call
        answer_generation_chain = prompt_template | llm_answer_generator | StrOutputParser()
        llm_answer_raw = answer_generation_chain.invoke({
            "user_query": input_data["user_query"],
            "context_str": input_data["context_str"]
        })
        return {**input_data, "llm_answer_raw": llm_answer_raw}


    def validate_llm_answer(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates the LLM's answer using a second LLM call, only if not skipped."""
        if input_data.get("skip_llm_and_validation"):
            return {**input_data, "validation_result": None, "is_valid": None} # No validation needed

        # If llm_answer is the fallback from prompt, no need to validate
        if input_data["llm_answer_raw"] == "I am not able to answer this question based on the context provided.":
             return {**input_data, "validation_result": None, "is_valid": True} # Considered valid as it followed instructions


        validation_chain = validation_prompt | llm_validator.with_structured_output(AnswerValidation)
        
        joined_context = input_data["context_str"] # Already formatted context
        
        validation_result = validation_chain.invoke({
            "user_query": input_data["user_query"],
            "joined_context_from_chunks": joined_context,
            "llm_answer": input_data["llm_answer_raw"]
        })
        return {**input_data, "validation_result": validation_result, "is_valid": validation_result.is_valid}

    def format_final_output(input_data: Dict[str, Any]) -> FinalOutput:
        """Formats the final output according to the specified schema."""
        user_query = input_data["user_query"] # For LangSmith logging
        llm_answer = input_data["llm_answer_raw"]
        retrieved_chunks = input_data["retrieved_chunks"]
        is_valid = input_data.get("is_valid") # Might be None if skipped

        chunks_used_output = [
            ChunkOutput(title=chunk.title, text=chunk.text, score=chunk.score)
            for chunk in retrieved_chunks
        ]
        
        manual_review = False
        final_answer_text = llm_answer

        if input_data.get("skip_llm_and_validation"): # No chunks case
            return FinalOutput(
                answer="I am not able to answer this question based on the context provided.",
                chunks_used=[],
                validated=None,
                manual_review_required=None
            )

        if is_valid is False:
            final_answer_text += "\n\n⚠️ This answer may reference information outside of your internal documentation. Manual review is recommended."
            manual_review = True
        
        # LangSmith metadata - this is a common pattern to add tags/metadata
        # For more granular control, one would use @traceable or context managers
        # However, with environment variables set, LangChain does a lot automatically.
        # To add custom metadata to the root trace, you might need to wrap the call.

        return FinalOutput(
            answer=final_answer_text,
            chunks_used=chunks_used_output,
            validated=is_valid,
            manual_review_required=manual_review
        )

    # Define the chain using RunnableLambda for custom logic steps
    # This LCEL chain structure ensures inputs/outputs flow correctly and are logged by LangSmith
    full_chain = (
        RunnableLambda(prepare_rag_input).with_config({"run_name": "PrepareRAGInput"})
        | RunnableLambda(generate_answer_if_chunks_exist).with_config({"run_name": "GenerateAnswer"})
        | RunnableLambda(validate_llm_answer).with_config({"run_name": "ValidateAnswer"})
        | RunnableLambda(format_final_output).with_config({"run_name": "FormatFinalOutput"})
    )
    return full_chain


# --- Main Execution ---
if __name__ == "__main__":
    print("System Initialized. LangSmith tracing should be active if configured correctly.")
    # Example Usage
    test_query = "What are the best practices for onboarding new clients?"
    
    print(f"User Query: {test_query}")

    # Create and invoke the RAG chain
    rag_chain = create_rag_chain()
    
    # LangSmith will automatically trace this invocation if configured
    # To add specific tags or metadata to the trace for this specific run:
    trace_id = uuid.uuid4() # Generate a unique ID for this run
    final_result = rag_chain.invoke(
        {"user_query": test_query},
        config={"metadata": {"user_id": "example_user_123", "session_id": "session_abc"}, "tags": ["test_run"], "run_name": f"RAG_Query_Run_{trace_id}"}
    )

    print("\n--- Final Output ---")
    print(final_result.model_dump_json(indent=2))

    # You can also inspect the trace in LangSmith using the project name you set.
    # The run_name can help you find specific runs.

    # Example of no valid chunks
    # For this to work, ensure retrieve_and_filter_chunks returns an empty list
    # This might require a query that genuinely has no good matches or adjusting the threshold.
    # print("\n\n--- Example: No Valid Chunks ---")
    # no_chunk_query = "asdasdasdflkjaslkdfjalksdjflkasdjf" # A query unlikely to match anything
    # no_chunk_result = rag_chain.invoke(
    # {"user_query": no_chunk_query},
    # config={"metadata": {"user_id": "example_user_no_chunk", "session_id": "session_def"}, "tags": ["no_chunk_test"], "run_name": "RAG_NoChunk_Run"}
    # )
    # print(no_chunk_result.model_dump_json(indent=2))

    # Example of answer not found in context
    # This requires chunks to be found, but the LLM to (correctly) state it cannot answer.
    # print("\n\n--- Example: Answer Not In Context (but chunks found) ---")
    # To simulate this, you might need to:
    # 1. Have some documents retrieved.
    # 2. Ask a question that is *not* answerable by those specific documents.
    # query_not_in_docs = "What is the color of the sky on Mars?" # Assuming docs are about digital marketing
    # not_in_docs_result = rag_chain.invoke(
    # {"user_query": query_not_in_docs},
    # config={"metadata": {"user_id": "example_user_not_in_docs", "session_id": "session_ghi"}, "tags": ["not_in_docs_test"], "run_name": "RAG_NotInDocs_Run"}
    # )
    # print(not_in_docs_result.model_dump_json(indent=2))

    # Example of answer being validated as False (requires specific setup of chunks and query)
    # This is the hardest to deterministically trigger without carefully crafted test data.
    # print("\n\n--- Example: Answer Fails Validation ---")
    # query_for_bad_answer = "Give me financial advice." # Assuming docs are about marketing and LLM might overreach
    # bad_answer_result = rag_chain.invoke(
    # {"user_query": query_for_bad_answer},
    # config={"metadata": {"user_id": "example_user_bad_answer", "session_id": "session_jkl"}, "tags": ["bad_answer_test"], "run_name": "RAG_BadAnswer_Run"}
    # )
    # print(bad_answer_result.model_dump_json(indent=2)) 