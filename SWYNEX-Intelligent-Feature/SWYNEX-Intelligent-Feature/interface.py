"""
interface.py - simple web UI for the DataQA Chatbot.

    python interface.py                 # hybrid mode (downloads the model on first run)
    python interface.py --no-model      # rules-only, fully offline
    python interface.py --data my.csv   # use your own CSV

Also usable from a notebook:  from interface import build_ui; build_ui().launch()
"""
import argparse

import gradio as gr

import dataqa
from dataqa import DataQABot, ModelBackend, ModelUnavailableError

EXAMPLES = [
    "Which region generated the highest revenue?",
    "Which region generated the lowest revenue?",
    "What was the best-selling product?",
    "How much revenue did North generate?",
    "Which month had the lowest revenue?",
    "What was the average revenue per sale?",
    "Who is the CEO of the company?",          # failure case: should abstain
    "Why did the top region perform best?",    # failure case: should abstain
]

BADGES = {"rules": "Exact (pandas)", "model": "AI model (extracted)",
          "abstain": "Not sure (abstained)", "error": "Error"}


def make_bot(data_path: str, use_model: bool):
    df = dataqa.load_data(data_path)          # raises DataError with a readable message
    backend, banner = None, "Running in **rules-only** mode (no language model)."
    if use_model:
        backend = ModelBackend()
        try:
            backend.load()
            banner = f"Hybrid mode: exact rules + `{dataqa.MODEL_NAME}`."
        except ModelUnavailableError as exc:
            backend = None
            banner = f"Model unavailable, so running rules-only. Details: {exc}"
    return DataQABot(df, backend), banner


def build_ui(data_path: str = "data/sales_data.csv", use_model: bool = True):
    bot, banner = make_bot(data_path, use_model)

    def respond(question):
        a = bot.ask(question)
        conf = "-" if a.confidence is None else f"{a.confidence:.2f}"
        return a.text, BADGES.get(a.source, a.source), conf, a.note

    with gr.Blocks(title="DataQA Chatbot") as demo:
        gr.Markdown("# DataQA Chatbot\nAsk plain-English questions about the sales dataset.")
        gr.Markdown(banner)
        q = gr.Textbox(label="Your question", placeholder="Which region generated the highest revenue?")
        btn = gr.Button("Ask", variant="primary")
        ans = gr.Textbox(label="Answer", interactive=False)
        with gr.Row():
            src = gr.Textbox(label="How it was answered", interactive=False)
            conf = gr.Textbox(label="Confidence", interactive=False)
        note = gr.Textbox(label="Details", interactive=False)
        gr.Examples(EXAMPLES, inputs=q)
        btn.click(respond, q, [ans, src, conf, note])
        q.submit(respond, q, [ans, src, conf, note])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sales_data.csv")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = ap.parse_args()
    try:
        build_ui(args.data, not args.no_model).launch(share=args.share)
    except dataqa.DataError as exc:
        raise SystemExit(f"Data problem: {exc}")
