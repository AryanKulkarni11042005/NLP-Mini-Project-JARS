import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

import gradio as gr
import pandas as pd
import sentiment

MODELS = sentiment.available_models()
if not MODELS:
    raise SystemExit("No models found in model/. Run the export notebook first.")

CLASSES = sentiment.classes
COLUMNS = ["Model", "Prediction", "Confidence"] + [c.capitalize() for c in CLASSES] + ["Time (ms)"]


def run(text, model):
    if not text or not text.strip():
        return {}, "", "", pd.DataFrame(columns=COLUMNS)

    rows = sentiment.compare(text)
    table = pd.DataFrame(
        [[r["model"], r["label"], r["confidence"]] + [r["probs"][c] for c in CLASSES] + [r["ms"]]
         for r in rows],
        columns=COLUMNS,
    )

    chosen = next((r for r in rows if r["model"] == model), rows[0])
    if chosen.get("error"):
        return {}, chosen["script"], chosen["text_used"], table
    return chosen["probs"], chosen["script"], chosen["text_used"], table


examples = [
    ["बैटरी लाइफ बहुत बढ़िया है", MODELS[0]],
    ["ये फोन बहुत धीमा है", MODELS[0]],
    ["kimat men ye phone achha hai", MODELS[0]],
    ["camera bahut kharab hai", MODELS[0]],
    ["kimmat mein ye fon acha hai", MODELS[0]],
]

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(lines=3, label="Review (Hindi or Hinglish)"),
        gr.Dropdown(choices=MODELS, value=MODELS[0], label="Model shown in the main result"),
    ],
    outputs=[
        gr.Label(num_top_classes=len(CLASSES), label="Sentiment (selected model)"),
        gr.Textbox(label="Detected script"),
        gr.Textbox(label="Text sent to the models (after normalization for Hinglish)"),
        gr.Dataframe(label="All models on this line", interactive=False),
    ],
    examples=examples,
    title="Hindi / Hinglish Review Sentiment",
    description="Pick a model from the dropdown. The table at the bottom always compares every exported model on the same line. English is not supported.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()