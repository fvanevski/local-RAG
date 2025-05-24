# llamaindex_rag_gui.py – Fully‑local RAG with vLLM or Ollama + Gradio
"""
A Gradio-based GUI for a fully local Retrieval Augmented Generation (RAG) system.
It supports document uploads (TXT, HTML, MD, PDF, DOCX) and URL ingestion,
builds a local vector index, and allows users to chat with an LLM (either vLLM or Ollama)
using the indexed documents as context.

Features
--------
* 📂 **Multi‑format queue** – supports .txt, .html, .md, .pdf, .docx plus URLs.
* ⏩ **Explicit “Process queue” button** – ingestion runs on demand.
* 🗄️ **State‑managed queue** – uploads and URLs are queued until processed.
* 🔄 **Incremental indexing** – new documents are added to the existing index.
* 🏷️ **Rich metadata** – chunks include `file_path` or `source_url`.
* 🛡 **Improved readers & validation** – uses LlamaIndex readers with fallbacks.
* 🖥️ **Cleaner UX** – drag‑and‑drop uploads, queue counter, clear button, token streaming.
* 🤖 **Flexible LLM Backend** – Supports vLLM (OpenAI-compatible server) or Ollama.

Requires
--------
Install necessary Python packages:
```bash
pip install gradio>=4 llama-index-core llama-index-readers-file llama-index-readers-web llama-index-llms-vllm llama-index-llms-ollama llama-index-embeddings-huggingface sentence-transformers
```

Usage
-----
Run the script from the command line, specifying the LLM backend:
```bash
python gui.py --llm-backend <backend_choice>
```
Where `<backend_choice>` is either `vllm` or `ollama`. This argument is required.

Examples:
*   To use a vLLM server: `python gui.py --llm-backend vllm`
*   To use an Ollama server: `python gui.py --llm-backend ollama`

Configuration
-------------
The script is configured primarily via environment variables. Default values are used if an environment variable is not set, but some are required based on the chosen LLM backend.

**Required Environment Variables (based on `--llm-backend` choice):**
*   If `--llm-backend vllm` is chosen:
    *   `VLLM_API_URL`: Must be set to the URL of your vLLM OpenAI-compatible server endpoint (e.g., `http://localhost:8000/v1`). There is no default if this backend is chosen; the variable must be present.
*   If `--llm-backend ollama` is chosen:
    *   `OLLAMA_API_URL`: Must be set to the URL of your Ollama server endpoint (e.g., `http://localhost:11434`). There is no default if this backend is chosen; the variable must be present.

**Optional Environment Variables (defaults are provided):**
*   `LLM_MODEL_NAME`: The name/identifier of the language model.
    *   Default: `mistralai/Mistral-7B-Instruct-v0.1`
    *   **For vLLM:** This should be a HuggingFace model identifier that your vLLM server is configured to serve (e.g., "mistralai/Mistral-7B-Instruct-v0.1").
    *   **For Ollama:** This should be an Ollama model name (e.g., "mistral", "llama3:8b-instruct-q5_K_M"). Ensure the model is available in your Ollama instance.
*   `EMBED_MODEL`: The HuggingFace model name for generating embeddings.
    *   Default: `sentence-transformers/all-MiniLM-L6-v2`
*   `CHUNK_SIZE`: Size of text chunks for parsing documents.
    *   Default: `512`
*   `CHUNK_OVERLAP`: Overlap between text chunks.
    *   Default: `64`
*   `MEMORY_TOKEN_LIMIT`: Token limit for chat memory. Also used as `context_window` for Ollama.
    *   Default: `2048`
*   `TOP_K`: Number of top similar chunks to retrieve for RAG.
    *   Default: `4`

Ensure that the chosen LLM server (vLLM or Ollama) is running and accessible at the specified URL, and that the `LLM_MODEL_NAME` is appropriate for that backend.
"""

import os
import argparse # Added for CLI argument parsing
import sys # Added for sys.exit
from pathlib import Path
from typing import List, Tuple
import shutil # Added for file copying

import gradio as gr
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader, DocxReader, HTMLTagReader, MarkdownReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.vllm import VLLM # Added VLLM import
from llama_index.core.llms import ChatMessage, MessageRole # Added for history conversion

# ---------- CONFIG ---------- #
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"

# Environment variable DEFAULTS
VLLM_API_URL_DEFAULT = "http://localhost:8000/v1"
OLLAMA_API_URL_DEFAULT = "http://localhost:11434"
LLM_MODEL_NAME_DEFAULT = "mistralai/Mistral-7B-Instruct-v0.1" # Suitable for vLLM by default
EMBED_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_DEFAULT = 512
CHUNK_OVERLAP_DEFAULT = 64
MEMORY_TOKEN_LIMIT_DEFAULT = 2048
TOP_K_DEFAULT = 4

# Global variable to store command-line arguments
ARGS = None

# These will be configured in _setup_llm_and_settings
# Settings.llm, Settings.embed_model, Settings.node_parser, Settings.callback_manager

def _parse_cli_args():
    """Parses command-line arguments for LLM backend selection."""
    parser = argparse.ArgumentParser(description="Local RAG Assistant with selectable LLM backend.")
    parser.add_argument(
        "--llm-backend",
        choices=["vllm", "ollama"],
        required=True,
        help="Choose the LLM backend: 'vllm' for a vLLM OpenAI-compatible server, or 'ollama' for an Ollama server."
    )
    return parser.parse_args()

def _setup_llm_and_settings(cli_args):
    """Initializes LLM and LlamaIndex settings based on CLI arguments and environment variables."""
    llm = None # Initialize llm to None
    backend_choice = cli_args.llm_backend

    # Retrieve model name and API URLs from environment variables, using defaults if not set.
    # The actual check for presence of API URL for the *selected* backend happens below.
    llm_model_name = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME_DEFAULT)
    
    if backend_choice == "vllm":
        vllm_api_url = os.getenv("VLLM_API_URL") # Check actual presence
        if not vllm_api_url:
            print("Error: --llm-backend set to 'vllm', but VLLM_API_URL environment variable is not set.", file=sys.stderr)
            sys.exit(1)
        print(f"Using VLLM backend. Model: {llm_model_name}, API URL: {vllm_api_url}")
        print(f"Ensure LLM_MODEL_NAME ('{llm_model_name}') is a HuggingFace model path compatible with your vLLM server.")
        llm = VLLM(
            model=llm_model_name,
            api_url=vllm_api_url,
            request_timeout=120.0
        )
    elif backend_choice == "ollama":
        ollama_api_url = os.getenv("OLLAMA_API_URL") # Check actual presence
        if not ollama_api_url:
            print("Error: --llm-backend set to 'ollama', but OLLAMA_API_URL environment variable is not set.", file=sys.stderr)
            sys.exit(1)
        print(f"Using Ollama backend. Model: {llm_model_name}, API URL: {ollama_api_url}")
        print(f"Ensure LLM_MODEL_NAME ('{llm_model_name}') is an Ollama model name (e.g., 'mistral', 'llama2:13b').")
        llm = Ollama(
            model=llm_model_name,
            base_url=ollama_api_url,
            context_window=int(os.getenv("MEMORY_TOKEN_LIMIT", MEMORY_TOKEN_LIMIT_DEFAULT)), # Ollama uses context_window
            is_chat_model=True, # Assuming chat model usage
            request_timeout=120.0
        )
    else:
        # This case should be caught by argparse `choices`
        print(f"Error: Invalid --llm-backend choice: {backend_choice}", file=sys.stderr)
        sys.exit(1)

    # Configure LlamaIndex Settings
    Settings.llm = llm
    
    embed_model_name = os.getenv("EMBED_MODEL", EMBED_MODEL_DEFAULT)
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    
    chunk_size = int(os.getenv("CHUNK_SIZE", CHUNK_SIZE_DEFAULT))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", CHUNK_OVERLAP_DEFAULT))
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    Settings.callback_manager = CallbackManager([LlamaDebugHandler()])
    
    # Create DOCS_DIR and INDEX_DIR if they don't exist
    # Moved here to ensure they are created after initial CLI parsing and basic setup
    DOCS_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)


# ---------- INDEX UTIL ---------- #

def load_or_create_index() -> VectorStoreIndex:
    if any(INDEX_DIR.iterdir()):
        storage_ctx = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
        return load_index_from_storage(storage_ctx)
    new_index = VectorStoreIndex([])
    new_index.storage_context.persist(persist_dir=str(INDEX_DIR))
    return new_index

index = load_or_create_index()
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=MEMORY_TOKEN_LIMIT)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=chat_memory,
    similarity_top_k=TOP_K,
    system_prompt=""  # Explicitly set system_prompt to empty string
)

# ---------- READERS ---------- #
pdf_reader = PDFReader()
docx_reader = DocxReader()
html_file_reader = HTMLTagReader() # For local HTML files
md_reader = MarkdownReader()
url_reader = BeautifulSoupWebReader() # For URLs

# ---------- FILE PROCESSING HELPER ---------- #

def _file_to_docs(tmp_upload_path: Path) -> List[Document]:
    """
    Processes a single temporary file uploaded by the user via Gradio.
    1. Copies the temporary file to a persistent local directory (`DOCS_DIR`).
    2. Selects the appropriate LlamaIndex file reader based on the file extension.
    3. Loads the file content into LlamaIndex `Document` objects.
    4. Assigns `file_path` (and `page_label` for PDFs) metadata to each document.
    5. Handles errors gracefully, providing user feedback via `gr.Error` or `gr.Warning`.
    Returns a list of Document objects, or an empty list if processing fails.
    """
    docs: List[Document] = []
    # Use the name of the temporary file, which Gradio usually sets to the original filename
    original_filename = tmp_upload_path.name 
    dest_path = DOCS_DIR / original_filename

    try:
        # Ensure DOCS_DIR exists (though it's created at startup, good for robustness)
        DOCS_DIR.mkdir(exist_ok=True)
        
        # Copy the uploaded file to our persistent DOCS_DIR
        shutil.copy(tmp_upload_path, dest_path)

        ext = dest_path.suffix.lower()
        
        loaded_docs_from_file: List[Document] = []
        if ext == ".pdf":
            loaded_docs_from_file = pdf_reader.load_data(dest_path)
            for i, d in enumerate(loaded_docs_from_file):
                d.metadata.setdefault("file_path", str(dest_path.name))
                # Add page number if the reader provides it, otherwise use index
                d.metadata.setdefault("page_label", d.metadata.get("page_label", i + 1))
        elif ext == ".docx":
            loaded_docs_from_file = docx_reader.load_data(dest_path)
            for d in loaded_docs_from_file:
                d.metadata.setdefault("file_path", str(dest_path.name))
        elif ext in [".html", ".htm"]:
            loaded_docs_from_file = html_file_reader.load_data(dest_path)
            for d in loaded_docs_from_file:
                d.metadata.setdefault("file_path", str(dest_path.name))
        elif ext in [".md", ".markdown"]:
            loaded_docs_from_file = md_reader.load_data(dest_path)
            for d in loaded_docs_from_file:
                d.metadata.setdefault("file_path", str(dest_path.name))
        elif ext == ".txt":
            text_content = dest_path.read_text(encoding="utf-8")
            doc = Document(text=text_content, metadata={"file_path": str(dest_path.name)})
            loaded_docs_from_file.append(doc)
        else:
            # Fallback: try to read as plain text
            try:
                text_content = dest_path.read_text(encoding="utf-8")
                doc = Document(text=text_content, metadata={"file_path": str(dest_path.name), "note": f"loaded as plain text (original type: {ext})"})
                loaded_docs_from_file.append(doc)
                gr.Warning(f"File '{original_filename}' (type '{ext}') was loaded as plain text.")
            except Exception as read_err:
                gr.Error(f"Unsupported file type '{ext}' for '{original_filename}'. Could not read as text: {read_err}")
                # If copy succeeded but read failed, dest_path might still exist.
                # Depending on desired behavior, could attempt to remove dest_path here.
                return [] # Return no docs for this file
        
        docs.extend(loaded_docs_from_file)

    except FileNotFoundError:
        gr.Error(f"Temporary uploaded file {tmp_upload_path} not found. It might have been deleted before processing.")
        return []
    except Exception as e:
        gr.Error(f"Error processing file '{original_filename}': {e}")
        # If copy or read failed, dest_path might not exist or be incomplete.
        return []
    
    return docs

# ---------- QUEUE HELPERS ---------- #

doc_queue_state: List[Document] = [] # Holds Document objects ready for processing

def _urls_to_docs(urls: List[str]) -> List[Document]:
    all_docs: List[Document] = []
    if not urls:
        return all_docs

    for url in urls:
        try:
            # BeautifulSoupWebReader().load_data() expects a list of URLs
            # So, we process one URL at a time to isolate errors and correctly attribute metadata
            # It returns a list of Document objects for each URL.
            documents_from_url = url_reader.load_data(urls=[url]) # Pass as a list
            
            for doc in documents_from_url:
                # Ensure metadata includes the source URL
                doc.metadata.setdefault("source_url", url) 
                # Add other relevant metadata if available/easy to extract
            all_docs.extend(documents_from_url)
            gr.Info(f"Successfully fetched and processed URL: {url}")
        except Exception as e:
            gr.Warning(f"⚠️ Failed to load URL '{url}': {e}")
            # Optionally, create a placeholder document indicating failure, or just skip
            # For now, just skipping and warning is fine.
            
    return all_docs

# Helper function to convert Gradio chat history to LlamaIndex ChatMessage objects
def convert_gradio_history_to_chatmessages(gradio_history: List[List[str]]) -> List[ChatMessage]:
    messages: List[ChatMessage] = []
    if not gradio_history:
        return messages
    for user_msg_str, bot_msg_str in gradio_history:
        if user_msg_str is not None:
            messages.append(ChatMessage(role=MessageRole.USER, content=str(user_msg_str)))
        
        # If bot_msg_str is not None, it means this turn had a bot response slot in Gradio's history.
        # We must include an ASSISTANT message to maintain alternation, even if content is empty.
        if bot_msg_str is not None:
            messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=str(bot_msg_str)))
        # If bot_msg_str is None, it implies the history from Gradio might be malformed for a completed turn,
        # or it represents a user message without a corresponding bot reply yet.
        # Given stream_answer gets `chat_history` before the current `query`,
        # `bot_msg_str` should ideally always be present for historical turns.
    return messages

def add_files_to_queue(files: List[gr.File], current_queue: List[Document]) -> Tuple[List[Document], str]:
    queue = current_queue or []
    added = 0
    for f in files:
        tmp = Path(f.name)
        docs = _file_to_docs(tmp)
        queue.extend(docs)
        added += len(docs)
    return queue, f"📥 Queued {added} new chunk(s). Total in queue: {len(queue)}"


def add_urls_to_queue(url_text: str, queue: List[Document]):
    queue = queue or []
    urls = [u.strip() for u in url_text.splitlines() if u.strip()]
    if not urls:
        return queue, "⚠️ Enter at least one URL."
    docs = _urls_to_docs(urls) # This is the call
    queue.extend(docs)
    # The f-string uses len(docs) which is the number of Document objects (chunks)
    return queue, f"🔗 Queued {len(docs)} document chunk(s) from URLs. Total in queue: {len(queue)}"


def clear_queue(_: bool, queue: List[Document]):
    return [], "🗑️ Cleared the pending queue."


def process_queue(queue: List[Document]):
    if not queue:
        return [], "⚠️ Queue is empty. Nothing to process."
    nodes = Settings.node_parser.get_nodes_from_documents(queue)
    index.insert_nodes(nodes)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    return [], f"✅ Indexed {len(queue)} document chunk(s). Queue cleared."


async def stream_answer(query: str, chat_history: List[List[str]]):
    if not query.strip():
        # Yield current history if query is empty, and then an empty string for the input box
        # This ensures the UI updates correctly without making an LLM call
        yield "", chat_history
        return

    # Convert Gradio history and set it to the chat engine's memory
    # This synchronizes the engine's memory with Gradio's displayed history for each call.
    # ChatMemoryBuffer.set() is a public method that replaces all messages in the buffer.
    # Accessing chat_engine.memory (the ChatMemoryBuffer instance) and calling .set()
    # is the appropriate way to manage history in a stateless-per-call Gradio setup.
    llm_chat_messages = convert_gradio_history_to_chatmessages(chat_history)
    chat_engine.memory.set(llm_chat_messages) 

    # astream_chat will use the history now set in chat_engine.memory,
    # add the new user query, process, and then the memory buffer within the
    # engine will be updated by LlamaIndex to include this latest turn.
    # However, since Gradio sends the full history each time, we overwrite it above
    # to ensure perfect sync with the UI's view of history.
    response_stream = await chat_engine.astream_chat(query)
    accumulated_response_content = ""

    # Stream tokens for the response
    async for delta in response_stream.async_response_gen(): # Corrected to use async_response_gen()
        # Ensure delta.content is not None before appending
        accumulated_response_content += (delta.content or "")
        
        # Yielding strategy for Gradio:
        # The function is a generator for a gr.Chatbot.
        # It yields a tuple: (value_for_user_query_box, value_for_chatbot_display).
        # Here, we send an empty string to clear the user_query_box,
        # and the progressively updated chat_history to the chatbot.
        # Gradio's Chatbot expects the full history list to be re-rendered on each yield.
        # chat_history is the history *before* this current query.
        # We append the current query and the so-far-accumulated response to it.
        yield "", chat_history + [[query, accumulated_response_content]]

    # After stream is complete, finalize the message with sources
    final_bot_message = accumulated_response_content
    source_nodes = response_stream.source_nodes
    if source_nodes:
        final_bot_message += "\\n\\n**Sources:**\\n"
        for sn in source_nodes:
            source_ref = sn.metadata.get('file_path') or sn.metadata.get('source_url') or 'Unknown Source'
            final_bot_message += f"- {source_ref} (score: {sn.score:.2f})\\n"

    # Update chat history for Gradio state
    # Gradio's `gr.Chatbot` expects the history to be a list of [user_msg, bot_msg] pairs.
    # The `chat_history` input to this function is the history *before* the current query.
    updated_history = chat_history + [[query, final_bot_message]]

    # Clear user input box and yield final state of chat history
    yield "", updated_history


# ---------- INTERFACE ---------- #
with gr.Blocks(title="Local RAG Assistant") as demo:
    gr.Markdown(
        """# 🤖 Local RAG Assistant  
**Important:** Start this script using a command-line flag to select your LLM backend:  
`python gui.py --llm-backend vllm` or `python gui.py --llm-backend ollama`  
Ensure `VLLM_API_URL` (for vLLM) or `OLLAMA_API_URL` (for Ollama) is set in your environment.  

Upload documents (TXT, PDF, DOCX, HTML, MD) or URLs, then click **Process queued docs** to build your private knowledge base.
"""
    )

    doc_queue = gr.State([])  # List[Document]

    with gr.Accordion("Ingest data", open=False):
        with gr.Tab("Upload Docs"):
            file_in = gr.File(
                file_types=[
                    ".txt",
                    ".html",
                    ".md",
                    ".pdf",
                    ".docx",
                ],
                file_count="multiple",
            )
            queue_status = gr.Markdown()
            file_in.upload(add_files_to_queue, inputs=[file_in, doc_queue], outputs=[doc_queue, queue_status])
        with gr.Tab("Add URLs"):
            url_box = gr.Textbox(lines=2, placeholder="One URL per line")
            add_url_btn = gr.Button("Add to queue")
            add_url_btn.click(add_urls_to_queue, inputs=[url_box, doc_queue], outputs=[doc_queue, queue_status])

        with gr.Row():
            proc_btn = gr.Button("🚀 Process queued docs")
            clear_btn = gr.Button("🗑️ Clear queue")
        proc_btn.click(process_queue, inputs=doc_queue, outputs=[doc_queue, queue_status])
        clear_btn.click(clear_queue, inputs=[clear_btn, doc_queue], outputs=[doc_queue, queue_status])

    chatbot = gr.Chatbot([], label="Chat", type="messages")
    user_query = gr.Textbox(label="Your question", placeholder="Ask something…")
    send_btn = gr.Button("Send")

    send_btn.click(stream_answer, inputs=[user_query, chatbot], outputs=[user_query, chatbot])

if __name__ == "__main__":
    ARGS = _parse_cli_args()
    
    # Setup LLM and other LlamaIndex settings based on CLI args
    _setup_llm_and_settings(ARGS) 
    
    # Initialize index and chat_engine after settings are configured
    # These were global before, ensure they are initialized after setup
    index = load_or_create_index()
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=int(os.getenv("MEMORY_TOKEN_LIMIT", MEMORY_TOKEN_LIMIT_DEFAULT))
    )
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=chat_memory,
        similarity_top_k=int(os.getenv("TOP_K", TOP_K_DEFAULT)),
        system_prompt="" 
    )
    
    demo.launch()
