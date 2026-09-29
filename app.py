import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)
import gradio as gr
from sentiment import predict
def run(text):
    if not text or not text.strip():
        return {}, "", ""
    r = predict(text)
    return r["probs"], r["script"], r["text_used"]

examples = [
    ["बैटरी लाइफ बहुत बढ़िया है"],
    ["ये फोन बहुत धीमा है"],
    ["kimat men ye phone achha hai"],
    ["camera bahut kharab hai"],
]

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=3, label="Review (Hindi or Hinglish)"),
    outputs=[
        gr.Label(num_top_classes=3, label="Sentiment"),
        gr.Textbox(label="Detected script"),
        gr.Textbox(label="Text sent to the model (after normalization for Hinglish)"),
    ],
    examples=examples,
    title="Hindi / Hinglish Review Sentiment",
    description="IndicBERT fine-tuned on Hindi and Hinglish product reviews. English is not supported.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=True)