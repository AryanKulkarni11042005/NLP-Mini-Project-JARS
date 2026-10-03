import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

import gradio as gr
import pandas as pd

import sentiment
from model_recommender import (
    analyze_dataset,
    detect_text_column,
    detect_label_column,
    detect_task,
    recommend_from_dataset,
)

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
        [
            [r["model"], r["label"], r["confidence"]]
            + [r["probs"][c] for c in CLASSES]
            + [r["ms"]]
            for r in rows
        ],
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


# ---------------------------------------------------------------------
# Dataset analyzer
# ---------------------------------------------------------------------

def get_columns(file):
    if not file:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)

    try:
        df = pd.read_csv(file)
        text_col = detect_text_column(df)
        label_col = detect_label_column(df, text_col)

        return (
            gr.update(choices=list(df.columns), value=text_col),
            gr.update(choices=["(None)"] + list(df.columns), value=label_col or "(None)"),
        )
    except Exception:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)


def _profile_rows(profile):
    rows = [
        ["Documents", profile["documents"]],
        ["Average characters", profile["avg_characters"]],
        ["Average tokens", profile["avg_tokens"]],
        ["Vocabulary size", profile["vocabulary_size"]],
        ["Unique token ratio (%)", profile["unique_token_ratio"]],
        ["Duplicate documents (%)", profile["duplicate_percentage"]],
        ["Emoji count", profile["emoji_count"]],
        ["Punctuation density (%)", profile["punctuation_density"]],
        ["Primary language", profile["primary_language"]],
        ["Code-mixed (%)", profile["code_mixed_percentage"]],
        ["Romanized (%)", profile["romanized_percentage"]],
    ]
    return pd.DataFrame(rows, columns=["NLP Metric", "Value"])


def _distribution_df(d, name):
    return pd.DataFrame(
        [{name: k, "Percentage": v} for k, v in d.items()]
    )


def analyze_uploaded_dataset(file, text_column, label_column, priority):
    empty = pd.DataFrame()

    if not file:
        return (
            "Upload a CSV dataset.",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            "No recommendation yet.",
            "",
        )

    try:
        df = pd.read_csv(file)

        if not text_column or text_column not in df.columns:
            text_column = detect_text_column(df)

        if label_column == "(None)":
            label_column = None
        elif label_column not in df.columns:
            label_column = detect_label_column(df, text_column)

        result = recommend_from_dataset(
            df,
            text_column=text_column,
            label_column=label_column,
            priority=priority,
        )

        profile = result["profile"]
        task = result["task"]
        recommendation = result["recommendation"]
        reason = result["reason"] or ""

        summary = (
            f"**Dataset:** {len(df):,} rows  |  "
            f"**Task:** {task}  |  "
            f"**Text column:** `{result['text_column']}`  |  "
            f"**Label column:** `{result['label_column'] or 'None'}`  |  "
            f"**Primary language:** {profile['primary_language']}"
        )

        if result["benchmark"] is not None:
            benchmark = result["benchmark"].copy()
            ranking = result["ranking"].copy()

            ranking_display = ranking[
                [
                    "Rank", "Model", "Accuracy", "Precision", "Recall",
                    "Macro F1", "Weighted F1", "Avg Time (ms)",
                    "Recommendation Score"
                ]
            ]

            lang_df = _distribution_df(profile["language_distribution"], "Language")
            script_df = _distribution_df(profile["script_distribution"], "Script")

            return (
                summary,
                _profile_rows(profile),
                lang_df,
                script_df,
                benchmark,
                ranking_display,
                lang_df,
                script_df,
                f"### Recommended model: {recommendation}\n\n{reason}",
                "\n".join(
                    f"- **{x[0]}**: {x[1]}"
                    for x in profile["top_tokens"]
                ),
            )

        lang_df = _distribution_df(profile["language_distribution"], "Language")
        script_df = _distribution_df(profile["script_distribution"], "Script")

        return (
            summary,
            _profile_rows(profile),
            lang_df,
            script_df,
            empty,
            empty,
            lang_df,
            script_df,
            f"### Recommendation: {recommendation}\n\n{reason}",
            "\n".join(f"- **{x[0]}**: {x[1]}" for x in profile["top_tokens"]),
        )

    except Exception as e:
        return (
            f"Error: {e}",
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            "Analysis failed. Check the CSV and selected columns.",
            "",
        )


demo = gr.Blocks()

with demo:
    gr.Markdown(
        """
# Hindi / Hinglish Sentiment + NLP Dataset Analyzer

Use the first section for individual sentiment prediction.

Use **Dataset Model Recommendation** to upload a CSV. The analyzer automatically profiles the NLP dataset,
detects language/script/code-mixing, identifies sentiment labels when available, benchmarks the four
already-trained models, ranks them according to your priority, and recommends a model.

**Important:** the benchmark never retrains the models.
"""
    )

    with gr.Tab("Single Review"):
        with gr.Row():
            with gr.Column():
                review_text = gr.Textbox(
                    lines=3,
                    label="Review (Hindi or Hinglish)",
                )
                selected_model = gr.Dropdown(
                    choices=MODELS,
                    value=MODELS[0],
                    label="Model shown in the main result",
                )
                predict_btn = gr.Button("Analyze Sentiment", variant="primary")

            with gr.Column():
                sentiment_output = gr.Label(
                    num_top_classes=len(CLASSES),
                    label="Sentiment (selected model)",
                )
                detected_script = gr.Textbox(label="Detected script")
                normalized_text = gr.Textbox(
                    label="Text sent to the models (after normalization for Hinglish)"
                )

        comparison_table = gr.Dataframe(
            label="All models on this line",
            interactive=False,
        )

        predict_btn.click(
            run,
            inputs=[review_text, selected_model],
            outputs=[
                sentiment_output,
                detected_script,
                normalized_text,
                comparison_table,
            ],
        )

        gr.Examples(
            examples=examples,
            inputs=[review_text, selected_model],
        )

    with gr.Tab("Dataset Model Recommendation"):
        gr.Markdown(
            """
### Upload a labeled or unlabeled NLP dataset

For a **labeled sentiment dataset**, the system measures all available existing models using:
**Accuracy, Macro Precision, Macro Recall, Macro-F1, Weighted-F1 and average inference time.**

For an **unlabeled dataset**, it reports the NLP profile but does not fabricate accuracy/F1 values.
"""
        )

        dataset_file = gr.File(
            label="Upload CSV dataset",
            file_types=[".csv"],
            type="filepath",
        )

        with gr.Row():
            text_column = gr.Dropdown(
                choices=[],
                label="Text column",
                interactive=True,
            )
            label_column = gr.Dropdown(
                choices=["(None)"],
                value="(None)",
                label="Label column (optional)",
                interactive=True,
            )

        priority = gr.Radio(
            choices=["Accuracy", "Speed", "Balanced"],
            value="Balanced",
            label="Application priority",
        )

        analyze_btn = gr.Button(
            "Analyze Dataset & Recommend Model",
            variant="primary",
        )

        dataset_summary = gr.Markdown()

        with gr.Row():
            profile_table = gr.Dataframe(
                label="NLP Dataset Profile",
                interactive=False,
            )
            language_table = gr.Dataframe(
                label="Language Distribution",
                interactive=False,
            )
            script_table = gr.Dataframe(
                label="Script Distribution",
                interactive=False,
            )

        gr.Markdown("## Model Benchmark")
        benchmark_table = gr.Dataframe(
            label="Measured Model Performance",
            interactive=False,
        )

        gr.Markdown("## Model Ranking")
        ranking_table = gr.Dataframe(
            label="Recommendation Ranking",
            interactive=False,
        )

        with gr.Row():
            language_chart = gr.BarPlot(
                x="Language",
                y="Percentage",
                title="Language Distribution",
                tooltip=["Language", "Percentage"],
            )
            script_chart = gr.BarPlot(
                x="Script",
                y="Percentage",
                title="Script Distribution",
                tooltip=["Script", "Percentage"],
            )

        recommendation = gr.Markdown()
        top_tokens = gr.Markdown()

        dataset_file.change(
            get_columns,
            inputs=[dataset_file],
            outputs=[text_column, label_column],
        )

        analyze_btn.click(
            analyze_uploaded_dataset,
            inputs=[dataset_file, text_column, label_column, priority],
            outputs=[
                dataset_summary,
                profile_table,
                language_table,
                script_table,
                benchmark_table,
                ranking_table,
                language_chart,
                script_chart,
                recommendation,
                top_tokens,
            ],
        )

        # Re-run chart components through the same analysis function using
        # the tables as the source of truth is not required; BarPlot accepts
        # the DataFrame returned to its matching component when connected.
        # To keep compatibility across Gradio versions, charts are optional.
        # The ranking and distribution tables remain authoritative.

if __name__ == "__main__":
    demo.launch()
