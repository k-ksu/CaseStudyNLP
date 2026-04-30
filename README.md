# Tokenization for Code-Switched and Mixed-Language Text

**Case Study — NLP Course**

## Topic

How do different tokenization methods handle code-switched text, where multiple languages appear within the same sentence (e.g., "Hey, vamos to the store porque necesitamos milk")?

## Research Question

Do multilingual subword tokenizers (mBERT, XLM-R, mT5) outperform monolingual ones (BERT, BETO, GPT-2) on code-switched text, and does tokenization quality predict downstream NER performance?

## Methodology

### Datasets

| Dataset | Languages | Size | Source |
|---------|-----------|------|--------|
| CMU Hinglish DoG | Hindi-English (Romanized) | 2000 utterances | HuggingFace `cmu_hinglish_dog` |
| Synthetic CS Spanish-English | Spanish-English | 40 sentences | Hand-crafted (`data/cs_spanish_english.txt`) |
| WikiANN (EN, ES, HI) | Monolingual baselines | 500 each | HuggingFace `wikiann` |

### Tokenizers Compared

| Tokenizer | Type | Algorithm | Vocab Size |
|-----------|------|-----------|------------|
| bert-base-uncased | Monolingual (EN) | WordPiece | 30K |
| dccuchile/bert-base-spanish-wwm-uncased | Monolingual (ES) | WordPiece | 31K |
| openai-community/gpt2 | Monolingual (EN) | BPE | 52K |
| bert-base-multilingual-cased | Multilingual (104 langs) | WordPiece | 110K |
| xlm-roberta-base | Multilingual (100 langs) | SentencePiece | 250K |
| google/mt5-base | Multilingual (101 langs) | SentencePiece | 250K |

### Metrics

**Tokenization quality (Notebook 02):**
- **Fertility** — average subword tokens per word (lower = less fragmentation)
- **OOV Rate** — percentage of [UNK] tokens (lower = better)
- **Sequence-Length Inflation** — ratio of token count to word count
- **Vocabulary Coverage** — percentage of words kept as single tokens
- **Per-Language Fertility** — fertility computed separately for EN vs ES words in CS text

**Downstream task (Notebook 03):**
- **NER F1 Score** — macro F1 on Named Entity Recognition (WikiANN)
- **Cross-lingual transfer** — train on English only, evaluate on Spanish and Hindi

### Experiments

1. **Tokenizer Comparison** — 6 tokenizers × 5 datasets = 30 metric combinations
2. **Per-Language Asymmetry** — fertility for EN vs ES words within the same CS sentence
3. **Hybrid Tokenization** — language-ID-aware routing (BERT for EN words, BETO for ES words)
4. **NER Fine-tuning** — 3 models trained on English WikiANN, tested cross-lingually
5. **Correlation Analysis** — Pearson correlation between fertility and NER F1

## Key Results

### Tokenization Quality

| Tokenizer | Hinglish DoG Fertility | CS Spanish-Eng Fertility |
|-----------|----------------------|------------------------|
| **XLM-R** | **1.53** | **1.33** |
| mT5 | 1.62 | 1.63 |
| mBERT | 1.77 | 1.41 |
| BERT (EN) | 1.73 | 1.65 |
| GPT-2 (EN) | 1.85 | 1.70 |
| BETO (ES) | 1.92 | 1.44 |

GPT-2 on Devanagari Hindi: fertility **8.94** (catastrophic). BETO on Hindi: **82% OOV**.

### Per-Language Asymmetry (Spanish-English CS)

| Tokenizer | EN words | ES words | Gap |
|-----------|----------|----------|-----|
| GPT-2 | 1.50 | **3.21** | 2.1× |
| BERT | 1.40 | **2.63** | 1.9× |
| XLM-R | 1.25 | 1.63 | **1.3× (most balanced)** |

### NER Cross-Lingual Transfer

| Model | English F1 | Spanish F1 | Hindi F1 |
|-------|-----------|-----------|---------|
| BERT (EN) | 62.0% | 45.0% | 16.4% |
| **mBERT** | **71.3%** | **62.6%** | **58.0%** |
| XLM-R | 61.5% | 50.2% | 52.0% |

*Trained on 1000 English WikiANN examples, tested on 200 examples per language.*

**Pearson correlation** fertility vs F1: **r = −0.81** (strong negative)

## Conclusions

1. **Multilingual tokenizers are necessary for code-switched text.** XLM-R provides the lowest fertility, zero OOV, and most balanced cross-lingual treatment.
2. **Monolingual tokenizers create dangerous asymmetry** — one language is fragmented 2–3× more than the other in the same sentence.
3. **Script mismatch is catastrophic** — monolingual tokenizers on unseen scripts produce unusable output.
4. **Fertility predicts downstream performance** — without training a model, fertility alone can estimate cross-lingual NER quality (r = −0.81).
5. **Hybrid tokenization adds complexity without clear benefit** — XLM-R matches or beats language-ID routing.

**Recommendation:** Use XLM-RoBERTa for any NLP pipeline processing code-switched or multilingual text.

## Project Structure

```
CaseStudyNLP/
├── README.md
├── requirements.txt
├── data/
│   └── cs_spanish_english.txt           # 40 hand-crafted Spanish-English CS sentences
├── notebooks/
│   ├── 01-data-exploration.ipynb        # EDA: datasets, language detection, text stats
│   ├── 02-tokenizer-comparison.ipynb    # Main experiment: 6 tokenizers × 5 datasets
│   └── 03-downstream-tasks.ipynb        # NER fine-tuning + cross-lingual transfer
├── src/
│   ├── __init__.py
│   └── utils.py                         # Metric and plotting utilities
├── figures/                             # PNG figures (saved directly from notebooks)
└── poster/                              # Poster PDF
```

## How to Run

```bash
pip install -r requirements.txt
```

Run notebooks in order: 01 → 02 → 03. Each notebook is self-contained and saves figures directly to `figures/`.

## Tools and Libraries

- **transformers** (HuggingFace) — tokenizers and models
- **datasets** (HuggingFace) — data loading
- **seqeval** — NER evaluation metrics
- **langid** — per-word language detection
- **matplotlib / seaborn** — visualization
- **pandas / numpy** — data processing
