# AW-MCTS: Adaptive-Width Monte Carlo Tree Search for Multi-hop Question Answering

> **Repository**: [https://github.com/llll0507/Adaptive-Width-Predictor](https://github.com/llll0507/Adaptive-Width-Predictor)

AW-MCTS is an advanced reasoning framework that combines Monte Carlo Tree Search (MCTS) with an adaptive width predictor for complex multi-hop question answering. The system dynamically predicts the appropriate MCTS search width for each question using a trained LDA model based on 10 text features, then decomposes questions, retrieves evidence, and synthesizes answers via risk-adaptive tree search.

## Features

- **Adaptive Width Prediction**: Uses LDA model with 10 text features (extracted via spaCy + VADER) to dynamically predict optimal MCTS search width (2–9) for each question, replacing fixed-width strategies.
- **Iterative Refinement through Decomposition**: Breaks down complex multi-hop questions into manageable sub-questions, enabling step-by-step reasoning.
- **Retrieval-Augmented Reasoning**: Augments LLM reasoning with TF-IDF based knowledge retrieval to incorporate external information.
- **Tree-based Exploration**: MCTS supports backtracking to mitigate error propagation in chain-of-thought reasoning.

## Project Structure

```
AW-MCTS-main/
├── evaluate_width_with_question.py      # Main experiment script (adaptive width + MCTS)
├── config.json                          # API keys and model configuration
├── requirements.txt                     # Python dependencies
├── data/
│   ├── demo.json                        # Test set for MCTS inference (MuSiQue format)
│   ├── train.csv                        # Training set for width predictor (10 features + labels)
│   └── test.csv                         # Test set for width predictor (10 features + labels)
├── train_models/
│   ├── train_models.py                  # Train RF/XGB/LDA width predictors
│   ├── extract_10_features.py           # Feature extraction utilities (extracts 10 features)
│   └── relaxed_acc_results/
│       ├── lda_width_predictor.pkl      # Trained LDA model checkpoint
│       └── relaxed_acc_results.json     # Training results summary
├── inference_analysis/
│   └── compare_fixed_vs_adaptive_musique.py  # Fixed vs adaptive width comparison
├── src/                                 # Core MCTS implementation
│   ├── task.py                          # MCTSTask definition and execution
│   ├── mcts.py                          # Monte Carlo Tree Search algorithm
│   ├── base.py                          # Base reasoning functions
│   └── node.py                          # Tree node definition
└── utils/                               # Utility modules
    ├── inference_model.py               # LLM API inference wrapper
    ├── value_model.py                   # Value model (Qwen2.5-7B-Instruct)
    ├── value_function.py                # Value functions for MCTS nodes
    ├── rag.py                           # TF-IDF retrieval-augmented generation
    ├── prompts.py                       # Prompt templates
    ├── verify.py                        # Answer verification (EM & F1)
    ├── extract.py                       # Data extraction utilities
    └── wrap.py                          # Prompt wrapping utilities
```

---

## Dataset Information

### Data Sources

The width prediction dataset was constructed from three publicly available multi-hop QA benchmarks:

| Dataset | DOI | URL | Sample Size (used) |
|---------|-----|-----|-------------------|
| **HotpotQA** | [10.18653/v1/d18-1259](https://doi.org/10.18653/v1/d18-1259) | https://hotpotqa.github.io/ | Randomly sampled |
| **2WikiMultihopQA** | [10.18653/v1/2020.coling-main.580](https://doi.org/10.18653/v1/2020.coling-main.580) | https://github.com/momohara/2wiki-multihop-qa | Randomly sampled |
| **MuSiQue** | [10.1162/tacl_a_00475](https://doi.org/10.1162/tacl_a_00475) | https://github.com/google-research-datasets/Musique | Randomly sampled |

> **Note**: All datasets are used under their original licenses (CC BY 4.0 for HotpotQA & 2WikiMultihopQA; MIT for MuSiQue). Full citation details are provided in the [Citation](#citation) section.

### Dataset Construction & Annotation

We randomly sampled questions from the above datasets and executed **full-width MCTS (width = 2–9)** on each. For each question that yielded a correct answer:
- **Appropriate width** = the smallest width at which the answer was correct.
- **Optimal width** = the width with the highest retrieval F1 among all correct-widths; if tied, the smallest width was selected.

After filtering and annotation, we obtained **2,422 labeled samples** of the form `(question, appropriate_width, optimal_width)`.
- **Train set**: 1,801 samples (used for training width predictors)
- **Test set**: 621 samples (held-out for evaluation; **no question overlap** with train set)

The train/test split ensures **unbiased offline evaluation** of the width predictor.

### Files in `data/`

| File | Description | Features | Labels |
|------|-------------|----------|--------|
| `demo.json` | Test set for **MCTS inference** (MuSiQue format) | Raw question + gold answer | — |
| `train.csv` | Training set for width predictor | 10 text features (see below) + `suitable_width`, `optimal_width` | `appropriate_width`, `optimal_width` |
| `test.csv` | Test set for width predictor | Same 10 features | `appropriate_width`, `optimal_width` |

> **Important**: `train.csv` and `test.csv` contain **pre-extracted features**, not raw text. See [Data Preprocessing](#data-preprocessing) for feature definitions.
---

## Data Preprocessing

The 10-dimensional feature vector for each question is computed as follows:

| # | Feature | Computation Method | Tool |
|---|---------|-------------------|------|
| 1 | `length_ratio` | Word count of the question | spaCy |
| 2 | `num_entities` | Number of named entities | spaCy (`en_core_web_sm`) |
| 3 | `sentence_complexity` | Count of complex dependency relations (conj, relcl, ccomp, xcomp) | spaCy |
| 4 | `keyword_density` | Ratio of content words (NOUN, ADJ, PROPN) to total tokens | spaCy |
| 5 | `sentiment_strength` | Absolute compound sentiment score | VADER |
| 6 | `dependency_depth` | Max depth of parsed dependency tree | spaCy |
| 7 | `multi_entity_flag` | Binary: 1 if named entities > 2 | spaCy |
| 8 | `pronoun_density` | Ratio of pronouns to total tokens | spaCy |
| 9 | `verb_density` | Ratio of verbs to total tokens | spaCy |
| 10 | `question_word_flag` | Binary flags for question word type (what/how/why) | Rule-based |

> Feature extraction code: [`train_models/extract_10_features.py`](train_models/extract_10_features.py)

The raw questions and gold answers were **not modified** (no lemmatization, no lowercasing beyond standard spaCy pipeline).
No external knowledge base was used beyond the original datasets.

---

## Adaptive Width Predictor

The width predictor uses a trained **LDA (Linear Discriminant Analysis)** model to predict the optimal MCTS search width (2–9) for each input question based on the 10 features above.

**Performance** (on test set, 621 samples):

| Model | Strict Accuracy | ±1 Accuracy | Relaxed Accuracy |
|-------|----------------|-------------|------------------|
| Random Forest | 43.96% | 58.29% | 73.27% |
| XGBoost | 35.59% | 52.66% | 70.85% |
| **LDA** | **44.77%** | **57.33%** | **73.59%** |

> *Relaxed Accuracy*: prediction is correct if the predicted width is in the set of widths that can produce a correct answer for that question.

---

## Reproducibility Workflow

### Step 1: Install Environment

```bash
conda create -n awmcts python=3.10 -y
conda activate awmcts
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Key dependencies** (see `requirements.txt` for full list):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computation |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.6.1 | LDA, RF, TF-IDF |
| spacy | 3.8.11 | NLP feature extraction |
| xgboost | 3.1.3 | XGBoost classifier |
| vaderSentiment | 3.3.2 | Sentiment analysis |
| torch | 2.3.1+cu118 | Value model inference |
| transformers | 4.47.0 | LLM tokenizer/model |
| requests | 2.32.5 | API calls |
| tqdm | 4.66.5 | Progress bars |
| matplotlib | 3.10.8 | Visualization |

### Step 2: Configure API Keys

Edit `config.json`:

```json
{
  "MODEL_API_KEY": "your-model-api-key",
  "MODEL_BASE_URL": "-",
  "HUGGINGFACEHUB_API_TOKEN": "your-hf-token"
}
```

### Step 3: Train Width Predictor (optional)

```bash
cd train_models
python train_models.py
```

Trains RF/XGB/LDA on `data/train.csv`, evaluates on `data/test.csv`, saves best model to `relaxed_acc_results/lda_width_predictor.pkl`.

### Step 4: Run Main Experiment

```bash
python evaluate_width_with_question.py
```

This will:
1. Load the trained LDA width predictor
2. For each question in `data/demo.json`, predict the optimal width
3. Run MCTS with the predicted width
4. Output exact match accuracy and F1 scores

### Step 5: Run Comparison Experiment

```bash
python inference_analysis/compare_fixed_vs_adaptive_musique.py
```

Compares fixed-width MCTS (width=2,3,...,9) against adaptive-width (LDA-predicted) on the MuSiQue dataset.

---

## Configuration

Key parameters configurable in `MCTSTask` (via `src/task.py`):

| Parameter | Description |
|-----------|-------------|
| `time_limit` | Time limit for search (ms) |
| `iteration_limit` | Maximum MCTS iterations |
| `exploration_constant` | UCT exploration constant |
| `multihops` | Number of sub-queries per decomposition |
| `total_depth` | Total depth of the search tree |
| `temperature` | LLM sampling temperature |
| `run_mode` | Reasoning strategy (MCTS, zero-shot, etc.) |
| `value_mode` | Value function mode (risk, similarity, etc.) |

---

## Evaluation Metrics

- **Exact Match (EM)**: Whether the predicted answer exactly matches the gold answer
- **F1 Score**: Retrieval F1 between retrieved paragraphs and gold supporting paragraphs
- **Strict Accuracy** (for width predictor): predicted width == best width
- **±1 Accuracy** (for width predictor): |predicted width − best width| ≤ 1
- **Relaxed Accuracy** (for width predictor): predicted width ∈ correct widths set

---

## Citation

If you use this code or dataset, please cite the original datasets:

```bibtex
@inproceedings{
  author       = {Zhilin Yang and
                  Peng Qi and
                  Saizheng Zhang and
                  Yoshua Bengio and
                  William W. Cohen and
                  Ruslan Salakhutdinov and
                  Christopher D. Manning},
  editor       = {Ellen Riloff and
                  David Chiang and
                  Julia Hockenmaier and
                  Jun'ichi Tsujii},
  title        = {HotpotQA: {A} Dataset for Diverse, Explainable Multi-hop Question
                  Answering},
  booktitle    = {Proceedings of the 2018 Conference on Empirical Methods in Natural
                  Language Processing, Brussels, Belgium, October 31 - November 4, 2018},
  pages        = {2369--2380},
  publisher    = {Association for Computational Linguistics},
  year         = {2018},
  url          = {https://doi.org/10.18653/v1/d18-1259},
  doi          = {10.18653/V1/D18-1259},
  timestamp    = {Mon, 12 May 2025 15:27:33 +0200},
  biburl       = {https://dblp.org/rec/conf/emnlp/Yang0ZBCSM18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{
  author       = {Xanh Ho and
                  Anh{-}Khoa Duong Nguyen and
                  Saku Sugawara and
                  Akiko Aizawa},
  editor       = {Donia Scott and
                  N{\'{u}}ria Bel and
                  Chengqing Zong},
  title        = {Constructing {A} Multi-hop {QA} Dataset for Comprehensive Evaluation
                  of Reasoning Steps},
  booktitle    = {Proceedings of the 28th International Conference on Computational
                  Linguistics, {COLING} 2020, Barcelona, Spain (Online), December 8-13,
                  2020},
  pages        = {6609--6625},
  publisher    = {International Committee on Computational Linguistics},
  year         = {2020},
  url          = {https://doi.org/10.18653/v1/2020.coling-main.580},
  doi          = {10.18653/V1/2020.COLING-MAIN.580},
  timestamp    = {Fri, 06 Aug 2021 00:39:51 +0200},
  biburl       = {https://dblp.org/rec/conf/coling/HoNSA20.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


@article{
  author       = {Harsh Trivedi and
                  Niranjan Balasubramanian and
                  Tushar Khot and
                  Ashish Sabharwal},
  title        = {{\unicode{9835}} MuSiQue: Multihop Questions via Single-hop Question
                  Composition},
  journal      = {Trans. Assoc. Comput. Linguistics},
  volume       = {10},
  pages        = {539--554},
  year         = {2022},
  url          = {https://doi.org/10.1162/tacl\_a\_00475},
  doi          = {10.1162/TACL\_A\_00475},
  timestamp    = {Wed, 19 Jun 2024 17:28:03 +0200},
  biburl       = {https://dblp.org/rec/journals/tacl/TrivediBKS22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

---

## License

- **Code**: MIT License
- **HotpotQA & 2WikiMultihopQA**: CC BY 4.0
- **MuSiQue**: MIT License
