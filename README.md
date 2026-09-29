# NLP Mini Project: Robustness of NLP Models to Transliteration and Spelling Noise in Hindi Text

**Team**

| Name | Roll No. |
|---|---|
| Jyotsna Kasibhotla | 23102B0078 |
| Siddharth Metkari | 24102B2006 |
| Rishant Singh | 23102B0065 |
| Aryan Kulkarni | 23102B0072 |

## Overview

Many people type Hindi in Roman letters ("Hinglish") and spell the same word several ways (`accha`, `achha`, `acha`). Models trained on clean Devanagari text have never seen this kind of input. This project measures how much transliteration and spelling noise hurt Hindi text classifiers, which models cope best, and how much of the loss can be recovered by normalizing the text.

**Research question:** How do transliteration and spelling noise affect the performance of different NLP models on Hindi text classification, and can normalization recover the lost performance?

The repository also contains a small Gradio demo that classifies the sentiment of a Hindi or Hinglish product review.

## Datasets

| Dataset | Task | Size | Notes |
|---|---|---|---|
| Hindi product reviews (`OdiaGenAI/sentiment_analysis_hindi`, Hugging Face) | Sentiment: positive / neutral / negative | 2,497 reviews (1,997 train, 500 test) | Main dataset, used for all four models and the demo |
| Hindi emotion dataset (`hindi.csv`) | 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise | about 8,000 sentences | Second check for TF-IDF, LSTM and fastText |

Both datasets are in Devanagari. Duplicate sentences are removed before an 80/20 stratified split, so no sentence appears in both train and test.

## Method

```
Devanagari dataset -> clean + split (80/20)
        |
        +--> Devanagari text
        +--> Hinglish (rule-based transliteration)
                   |
                   +--> add spelling noise (Low / Medium / High) [test set only]
                                |
                                +--> normalize (edit-distance repair)
                                |
                                +--> 4 models --> macro F1
```

1. **Transliteration.** Devanagari is converted to Roman letters with `indic-transliteration` (ITRANS) plus rules for casual spelling (Devanagari dotted letters cleaned, silent final vowel dropped, lowercase). English loanwords come out phonetically (`tebalet` for "tablet") because Devanagari does not record the original spelling.
2. **Spelling noise.** Applied to the Hinglish test set only. Each word is edited with probability 0.3 (Low), 0.6 (Medium) or 1.0 (High). An edit is a letter swap (`aa` to `a`, `i` to `ee`, `kh` to `k`, `ch` to `chh`, `v` to `w`), a dropped letter, or a stretched short word. Results are averaged over 5 noise seeds.
3. **Normalization.** Each unknown word is mapped to the closest word in the training Hinglish vocabulary: candidates of similar length, difflib similarity of at least 0.75, then the most frequent among matches within 0.08 of the best score. The vocabulary comes from the training set only.
4. **Metric.** Macro F1, because the classes are imbalanced and accuracy is inflated by the largest class.

## Models

| Model | How it sees an unseen spelling |
|---|---|
| TF-IDF + Logistic Regression | Whole-word counts, so a misspelling is an unknown word |
| Bidirectional LSTM | Learns word embeddings from scratch, also needs exact words |
| fastText | Builds words from character pieces (n-grams of length 2 to 5) |
| IndicBERT (`ai4bharat/IndicBERTv2-MLM-only`), fine-tuned | Pretrained subword transformer for Indian languages |

All models use class-balanced training. Each was trained in up to three setups: Hindi only, Hinglish only, and Hindi + Hinglish together.

## Results (product review dataset)

Scores are macro F1 on 500 test reviews (70 negative), one training run per model. Differences under about 0.03 are within noise.

**Robustness on Hinglish text (models trained on Hinglish only)**

| Model | Clean Hinglish | High noise | Drop | High noise, normalized |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.696 | 0.251 | 64% | 0.654 |
| LSTM | 0.631 | 0.194 | 69% | 0.617 |
| fastText | 0.655 | 0.542 | 17% | 0.627 |
| IndicBERT | 0.674 | 0.493 | 27% | 0.660 |

**Clean Hindi (models trained on Hindi + Hinglish)**

| Model | Clean Hindi | Clean Hinglish |
|---|---|---|
| TF-IDF + LogReg | 0.686 | 0.694 |
| LSTM | 0.629 | 0.603 |
| fastText | 0.647 | 0.649 |
| IndicBERT | 0.832 | 0.719 |

**Main findings**

- A model trained only on Devanagari is at chance level on Hinglish (about 0.19 for three classes) for TF-IDF, LSTM and fastText. IndicBERT trained on Hindi only still reaches 0.493, so it transfers partly across scripts.
- Word-based models (TF-IDF, LSTM) lose about two thirds of their score at High noise, because a misspelled word becomes an unknown word. fastText loses the least (17%), which supports the idea that character pieces help against spelling variation.
- Normalization recovers most of the loss for every model. After normalization all four models score between 0.62 and 0.66 on Hinglish at High noise.
- IndicBERT is clearly best on clean Hindi (0.83 vs 0.63 to 0.69).
- On the 7-class emotion dataset (about 0.25 macro F1, with chance at 0.14), fastText also lost the least under noise, so the ordering agrees across both datasets.

## Limitations

- The noise and the normalizer are rules written by us. The normalizer is designed to undo exactly this kind of edit, so the recovery numbers are an upper bound. Real user spelling is messier (`bhot` for `bahut`).
- Hinglish is generated by rule-based transliteration, not collected from real users, and English loanwords are spelled phonetically.
- Frequency preference in the normalizer can overwrite rare words (for example `bajr` was mapped to `bar` instead of `bajar`).
- One training run per model on a small test set. Gaps under about 0.03 should not be read as real differences.
- English and real code-mixed text are not supported.

## Setup

Create the conda environment:

```bash
conda create --name nlp-mini-proj python=3.11
conda activate nlp-mini-proj
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Running the demo
Start the Gradio UI:

```bash
python app.py
```

Then open `http://127.0.0.1:7860` in your browser. To get a temporary public link, change the last line of `app.py` to `demo.launch(share=True)`.

**What the demo does:** you type a Hindi (Devanagari) or Hinglish (Roman) review. Hinglish input is normalized first, then IndicBERT predicts positive, neutral or negative with a confidence for each class. The interface also shows the detected script and the text actually sent to the model. English input is not supported.

## Tech stack

Python 3.11, scikit-learn, PyTorch, Hugging Face Transformers, fastText, indic-transliteration, Gradio.