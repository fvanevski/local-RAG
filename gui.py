# llamaindex_rag_gui.py – Fully‑local RAG with vLLM + Gradio (rev‑2)
"""
Changes in this revision
-----------------------
* 📂 **Multi‑format queue** – supports .txt, .html, .md, .pdf, .docx plus URLs, gathered in a queue that the user can grow over multiple batches.
* ⏩ **Explicit “Process queue” button** – ingestion runs only when the user clicks **Process queued docs**; index persists incrementally.
* 🗄️ **State‑managed queue** – uses a `gr.State` list of `Document` objects so uploads and URLs are merged until processed.
* 🔄 **Incremental indexing** – new nodes are inserted into the existing `VectorStoreIndex` without retracing previous docs.
* 🏷️ **Rich metadata** – each chunk carries `file_path` or `source_url` plus page numbers or headings when available.
* 🛡 **Improved readers & validation** – leverages LlamaIndex readers (`PDFReader`, `DocxReader`, `HTMLReader`, `MarkdownReader`) and graceful fallback for plain‑text.
* 🖥️ **Cleaner UX** – drag‑and‑drop multi‑file upload, live queue counter, “Clear queue” helper, and chat token streaming.

Requires  
```bash
pip install gradio>=4 llama‑index‑core llama‑index‑readers‑file llama‑index‑readers‑web llama‑index‑llms‑vllm
```
"""

import os
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
from llama_index.core.llms import ChatMessage, MessageRole # Added for history conversion

# ---------- CONFIG ---------- #
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"

LLM_API_BASE = os.getenv("VLLM_API", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "hf.co/mradermacher/Qwen1.5-4B-Chat-i1-GGUF:Q6_K")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_LIMIT", 2048))
TOP_K = int(os.getenv("TOP_K", 4))

DOCS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ---------- LLM & EMBEDDINGS ---------- #
llm = Ollama(
    model=LLM_MODEL_NAME,
    api_base=LLM_API_BASE,
    api_key="EMPTY",  # For vLLM or other OpenAI-compatible servers not requiring a key
    context_window=32768, # Explicitly set context window for Mistral-7B-Instruct-v0.2
    is_chat_model=True,
    # temperature=0.1, # Optional: Add other LiteLLM parameters if needed
    # max_tokens=512,   # Optional: Control output length
)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

callback_manager = CallbackManager([LlamaDebugHandler()])
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = parser
Settings.callback_manager = callback_manager

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
    Processes a single uploaded temporary file:
    1. Copies it to the persistent DOCS_DIR.
    2. Uses the appropriate LlamaIndex reader based on extension.
    3. Returns a list of LlamaIndex Document objects with metadata.
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
    docs = _urls_to_docs(urls)
    queue.extend(docs)
    return queue, f"🔗 Queued {len(urls)} page(s). Total in queue: {len(queue)}"


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
    # This synchronizes the engine's memory with Gradio's displayed history.
    llm_chat_messages = convert_gradio_history_to_chatmessages(chat_history)
    chat_engine._memory.set(llm_chat_messages) # Access the protected _memory attribute

    # astream_chat will use the synchronized memory, add the new user query,
    # process, and then update its memory with the new user query + AI response.
    response_stream = await chat_engine.astream_chat(query)
    accumulated_response_content = ""

    # Stream tokens for the response
    async for delta in response_stream.async_response_gen(): # Corrected to use async_response_gen()
        # Ensure delta.content is not None before appending
        accumulated_response_content += (delta.content or "")
        # Yield intermediate accumulated response for streaming effect
        current_chat_entry = [query, accumulated_response_content]
        # Update the last entry in chat_history or append if it's a new interaction
        if chat_history and chat_history[-1][0] == query:
            # This logic might be tricky if user sends same query twice;
            # Gradio usually handles history by appending.
            # For streaming, we update the *display* of the current turn.
            # The final history update happens after the loop.
            # For now, let's just yield the accumulated content for the current turn.
            # Gradio's gr.Chatbot handles displaying this progressively.
            yield "", chat_history[:-1] + [current_chat_entry] # Send updated history for display
        else:
            # This case might not be hit if Gradio handles history append before calling stream_answer
            # or if we always append then update.
            # Let's simplify: Gradio's Chatbot expects the full history + current streaming message.
            # The input `chat_history` is the history *before* this turn.
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
Upload **TXT / HTML / Markdown / PDF / DOCX** or paste URLs.  
Add as many as you like, then click **Process queued docs** to build your private knowledge base.  
All ranking & generation run **entirely on your machine** using vLLM + LlamaIndex.
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
    demo.launch()
