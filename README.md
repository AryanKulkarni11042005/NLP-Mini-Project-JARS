**# NLP Mini Project: Robustness of NLP Models to Transliteration and Spelling Noise in Hindi Text**
****Team****
| Name | Roll No. |
|---|---|
| Jyotsna Kasibhotla | 23102B0078 |
| Siddharth Metkari | 24102B2006 |
| Rishant Singh | 23102B0065 |
| Aryan Kulkarni | 23102B0072 |

**## Overview**
Many people type Hindi in Roman letters ("Hinglish") and spell the same word several ways (`accha`, `achha`, `acha`). Models trained on clean Devanagari text have never seen this kind of input. This project measures how much transliteration and spelling noise hurt Hindi text classifiers, which models cope best, and how much of the loss can be recovered by normalizing the text.

****Research question:**** How do transliteration and spelling noise affect the performance of different NLP models on Hindi text classification, and can normalization recover the lost performance?

The repository also contains a Gradio application for single-review sentiment prediction, uploaded-dataset analysis, and model evaluation.

**## Datasets**
| Dataset | Task | Size | Notes |
|---|---|---|---|
| Hindi product reviews (`OdiaGenAI/sentiment_analysis_hindi`, Hugging Face) | Sentiment: positive / neutral / negative | 2,497 reviews (1,997 train, 500 test) | Main dataset, used for all four models and the demo |
| Hindi emotion dataset (`hindi.csv`) | 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise | about 8,000 sentences | Second check for TF-IDF, LSTM and fastText |

Both datasets are in Devanagari. Duplicate sentences are removed before an 80/20 stratified split, so no sentence appears in both train and test.

**## Method**
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

1. ****Transliteration.**** Devanagari is converted to Roman letters with `indic-transliteration` (ITRANS) plus rules for casual spelling (Devanagari dotted letters cleaned, silent final vowel dropped, lowercase). English loanwords come out phonetically (`tebalet` for "tablet") because Devanagari does not record the original spelling.
2. ****Spelling noise.**** Applied to the Hinglish test set only. Each word is edited with probability 0.3 (Low), 0.6 (Medium) or 1.0 (High). An edit is a letter swap (`aa` to `a`, `i` to `ee`, `kh` to `k`, `ch` to `chh`, `v` to `w`), a dropped letter, or a stretched short word. Results are averaged over 5 noise seeds.
3. ****Normalization.**** Each unknown word is mapped to the closest word in the training Hinglish vocabulary: candidates of similar length, difflib similarity of at least 0.75, then the most frequent among matches within 0.08 of the best score. The vocabulary comes from the training set only.
4. ****Metric.**** Macro F1, because the classes are imbalanced and accuracy is inflated by the largest class.

**## Models**
| Model | How it sees an unseen spelling |
|---|---|
| TF-IDF + Logistic Regression | Whole-word counts, so a misspelling is an unknown word |
| Bidirectional LSTM | Learns word embeddings from scratch, also needs exact words |
| fastText | Builds words from character pieces (n-grams of length 2 to 5) |
| IndicBERT (`ai4bharat/IndicBERTv2-MLM-only`), fine-tuned | Pretrained subword transformer for Indian languages |

All models use class-balanced training. Each was trained in up to three setups: Hindi only, Hinglish only, and Hindi + Hinglish together.

Fine-Tuned indicBERT model can be found at this link: `https://drive.google.com/drive/folders/11Fyju7EzLF4X4iYmymkvlKJ8pHsIerjK?usp=sharing\`

**## Results (product review dataset)**
Scores are macro F1 on 500 test reviews (70 negative), one training run per model. Differences under about 0.03 are within noise.

****Robustness on Hinglish text (models trained on Hinglish only)****
| Model | Clean Hinglish | High noise | Drop | High noise, normalized |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.696 | 0.251 | 64% | 0.654 |
| LSTM | 0.631 | 0.194 | 69% | 0.617 |
| fastText | 0.655 | 0.542 | 17% | 0.627 |
| IndicBERT | 0.674 | 0.493 | 27% | 0.660 |

****Clean Hindi (models trained on Hindi + Hinglish)****
| Model | Clean Hindi | Clean Hinglish |
|---|---|---|
| TF-IDF + LogReg | 0.686 | 0.694 |
| LSTM | 0.629 | 0.603 |
| fastText | 0.647 | 0.649 |
| IndicBERT | 0.832 | 0.719 |

****Main findings****
- A model trained only on Devanagari is at chance level on Hinglish (about 0.19 for three classes) for TF-IDF, LSTM and fastText. IndicBERT trained on Hindi only still reaches 0.493, so it transfers partly across scripts.
- Word-based models (TF-IDF, LSTM) lose about two thirds of their score at High noise, because a misspelled word becomes an unknown word. fastText loses the least (17%), which supports the idea that character pieces help against spelling variation.
- Normalization recovers most of the loss for every model. After normalization all four models score between 0.62 and 0.66 on Hinglish at High noise.
- IndicBERT is clearly best on clean Hindi (0.83 vs 0.63 to 0.69).
- On the 7-class emotion dataset (about 0.25 macro F1, with chance at 0.14), fastText also lost the least under noise, so the ordering agrees across both datasets.

**## Gradio application**

The application uses a dark, responsive layout with a centered content area and five sections: **Home**, **Single Review**, **Dataset Analysis**, **Dashboard**, and **About**.

In **Single Review**, enter a Hindi or Hinglish review to see the selected model's sentiment prediction, confidence scores, detected script, and comparisons from the available models.

In **Dataset Analysis**, upload a CSV to profile the data and, when its labels support sentiment classification, benchmark the available pretrained models. Choose **Accuracy**, **Speed**, or **Balanced** to rank the models and see a recommendation. The analysis includes:

The analyzer provides:
- Automatic text and label column detection.
- Task detection for sentiment classification when compatible labels are available.
- Dataset statistics including document count, character length, token length, vocabulary size, unique token ratio and duplicate percentage.
- NLP-specific analysis including punctuation density, emoji count and top tokens.
- Automatic language detection for languages such as Hindi, Marathi and English.
- Script detection for Latin, Devanagari and mixed-script text.
- Detection of code-mixed and romanized text.
- Language and script distribution charts.

When labels are available for a compatible sentiment-classification dataset, the application benchmarks the existing models without retraining them and reports Accuracy, Precision, Recall, Macro F1, Weighted F1, and average inference time. It also shows text-transformation quality scores and robustness results across clean, noisy, and normalized text conditions. The **Dashboard** displays model-level evaluation summaries and charts.

The user can select:
- **Accuracy** — prioritizes measured classification quality.
- **Speed** — prioritizes inference speed.
- **Balanced** — combines measured model quality and inference speed.

The system then ranks the available existing models and provides a model recommendation based on the selected priority.

For unlabeled datasets or unsupported NLP tasks, the application reports dataset compatibility and profile information without fabricating accuracy or F1 results. Charts and tables are shown in the browser. Uploaded CSVs and plots are not saved as persistent project files; downloadable HTML and PDF reports are generated for the current analysis.

**## Limitations**
- The noise and the normalizer are rules written by us. The normalizer is designed to undo exactly this kind of edit, so the recovery numbers are an upper bound. Real user spelling is messier (`bhot` for `bahut`).
- Hinglish is generated by rule-based transliteration, not collected from real users, and English loanwords are spelled phonetically.
- Frequency preference in the normalizer can overwrite rare words (for example `bajr` was mapped to `bar` instead of `bajar`).
- One training run per model on a small test set. Gaps under about 0.03 should not be read as real differences.
- English and real code-mixed text are not supported.

## Evaluation Metrics

The project reports two separate kinds of evaluation. Text overlap scores never use sentiment labels; they compare generated or transformed text with an aligned reference. Classification scores use sentiment labels and model predictions.

### Classification metrics

- **Accuracy** is the fraction of labels predicted correctly; higher is better.
- **Precision** and **recall** are macro averages across sentiment classes; higher is better. Macro averaging gives minority classes equal weight.
- **Macro F1** is the harmonic mean of macro precision and recall. It is the primary robustness metric because the classes are imbalanced; higher is better.
- **Weighted F1** averages class F1 by class frequency; higher is better, but it can hide weak performance on rare classes.

The experiment compares clean Hindi, clean Hinglish, low/medium/high noise Hinglish, and normalized Hinglish at each noise level. Noise uses the existing five seeds. Reports include per-seed rows and means/sample standard deviations where multiple seeds exist. The high-noise absolute/percentage F1 drop and recovery after normalization are reported separately.

### Text-transformation metrics

The reference is each held-out clean Hinglish string generated by the existing deterministic transliterator from that same Hindi sentence. Candidates are the paired noisy or normalized Hinglish strings. This is a controlled reference for lexical preservation/restoration; the repository has no independent human Hinglish references, so these scores do **not** measure the quality of Hindi-to-Hinglish transliteration itself. The clean Hinglish condition is the identity baseline. No reference text is fabricated.

Tokenization is explicit and script preserving: a Unicode-category tokenizer groups letters, numbers, and combining marks (including Indic vowel signs) and keeps punctuation as separate tokens for BLEU, ROUGE, and METEOR; SacreBLEU is told to use the provided whitespace tokenization. ROUGE uses no English stemming. METEOR uses its alignment/fragmentation calculation with exact token matches only, without English Porter stemming or WordNet. chrF scores the raw text at the character level, without transliteration, lowercasing, or whitespace scoring.

- **BLEU** measures clipped n-gram precision with a brevity penalty; higher indicates greater overlap.
- **BLEU-1**, **BLEU-2**, **BLEU-3**, and **BLEU-4** report the BLEU calculation limited to that maximum n-gram order; higher is better. BLEU scores are corpus-level and use exponential smoothing/effective order for short sentences.
- **ROUGE-1** measures unigram overlap, **ROUGE-2** bigram overlap, and **ROUGE-L** longest-common-subsequence overlap. The reported values are mean sentence-level F-measures; higher is better.
- **METEOR** combines exact unigram matches with a penalty for fragmented alignments. Because no Hindi/Hinglish stemmer or synonym resource is used, it is a lexical score rather than a semantic one; higher is better.
- **chrF** measures character n-gram precision/recall (orders 1–6, beta 2). It is useful for spelling variation because partial character overlap is retained; higher is better.
- **TER** estimates the number of word edits required to turn a candidate into its reference, normalized by reference length. It is included as a complementary restoration measure; lower is better. Since the experiment preserves word order, TER’s shift operation is not expected to dominate.

These metrics are not sentiment-classification metrics and are not combined into an overall score. Their scales differ: BLEU/ROUGE/METEOR/chrF/TER are emitted on a 0–100 scale, while classification metrics use 0–1.

### Reproducible evaluation outputs

The reusable implementations are in `evaluation/`. The export notebook reuses the existing dataset split (`random_state=42`), transliteration, train-only normalizer vocabulary, corruption rates, and five noise seeds. It writes new run-level and mean/standard-deviation CSVs to `evaluation_outputs/` under the notebook’s export directory, plus plots in `evaluation_outputs/plots/`. Existing results are not overwritten; a numbered filename is used if an output already exists.

From the repository root, install the pinned dependencies in the Python 3.11 environment and open the export notebook:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebooks/NLP_Mini_Project_export.ipynb
```

The saved paired test set from `02_advance.ipynb` can be scored without training models or downloading data:

```bash
python -m evaluation.evaluate_saved_pairs --pairs eval_sets.pkl
```

This artifact belongs to the secondary Hindi emotion dataset and contains three existing noise seeds. The command writes text-metric run/summary CSVs and text-quality plots under `evaluation_outputs/`; it does not produce classification results because saved predictions/model artifacts are not present in this checkout.

The export notebook includes its original training cells. If trained model callables already exist in the notebook session, run the evaluation cells at the end without rerunning the training cells. The notebook evaluates IndicBERT only when its existing local artifact is available; it does not download model weights. Without that artifact, the report names only the predictors available in the session.

The text metrics are also covered by a small controlled test suite:

```bash
python -m unittest discover -s tests
```

The quality/F1 correlations are descriptive and exploratory. They reuse a single held-out corpus across conditions and seeds, so they do not establish statistical significance or independent-sample evidence.

**## Setup**
Create the conda environment:
```bash
conda create --name nlp-mini-proj python=3.11
conda activate nlp-mini-proj
```

Install requirements:
```bash
pip install -r requirements.txt
```

**## Running the demo**
Start the Gradio UI:
```bash
python app.py
```

Open the local URL printed in the terminal (normally `http://127.0.0.1:7860`). Stop any earlier app instance before restarting it if that port is already in use.

The included `nlp_multilingual_test_dataset.csv` is a small labeled dataset for trying the upload and evaluation workflow. Robustness analysis uses up to 500 deterministic sample rows. English input is not supported for single-review sentiment prediction.

**## Tech stack**
Python 3.11, scikit-learn, PyTorch, Hugging Face Transformers, fastText, indic-transliteration, Gradio, lingua-language-detector.
