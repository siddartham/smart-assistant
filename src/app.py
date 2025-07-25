import gradio as gr

from .graph import graph
from .indexer import load_and_index
from .logger import logger

config = {"configurable": {"thread_id": "session1"}}


def upload_docs(file_objs):
    if not file_objs or isinstance(file_objs, str):
        logger.warning("No files uploaded")
        return "⚠️ No files uploaded. Please upload at least one PDF document."
    file_paths = [file.name for file in file_objs]
    return load_and_index(file_paths)


def chat(user_input: str, history):
    logger.info(f"User Input: {user_input}")
    messages = [
        {"role": "system", "content": (
            "Always answer using tools. Do not guess. \n"
            "Choose tools carefully based on user intent. If the query includes summarization, insights, overviews, "
            "or key points, use the summarization tool.\n"
            "After answering, Explain:\n"
            "1. Why these tools were chosen in sequence\n"
            "2. How the final answer was derived\n"
            "3. What the source of truth is (e.g., document, structured tables)\n\n"
            "Format your response with headings: Answer, Tool Trace, Reasoning, Source"
        )
         }
        ,
        {"role": "user", "content": user_input}
    ]

    result = graph.invoke({"messages": messages}, config=config)
    return result["messages"][-1].content


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Smart Assistant")

    with gr.Row():
        with gr.Column():
            file_upload = gr.File(label="Upload PDFs", file_types=[".pdf", ".txt", ".docx"], file_count="multiple")
            upload_btn = gr.Button("Index Documents")
            upload_status = gr.Textbox(label="Upload Status")
            upload_btn.click(fn=upload_docs, inputs=file_upload, outputs=upload_status)

        with gr.Column():
            gr.Markdown("### 💬 Assistant Chat")
            chatbot = gr.ChatInterface(chat, type="messages")

demo.launch()
