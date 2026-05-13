# AW-MCTS: Adaptive-Width Monte Carlo Tree Search for Multi-hop Question Answering

> **Repository**: [https://github.com/llll0507/Adaptive-Width-Predictor](https://github.com/llll0507/Adaptive-Width-Predictor)

AW-MCTS is an advanced reasoning framework that combines Monte Carlo Tree Search (MCTS) with an adaptive width predictor for complex multi-hop question answering. The system dynamically predicts the optimal MCTS search width for each question using a trained LDA (Linear Discriminant Analysis) model based on 10 text features, then decomposes complex questions into sub-questions, retrieves relevant information, and synthesizes the final answer through iterative tree search with risk-adaptive evaluation.

## Features

- **Adaptive Width Prediction**: Uses LDA model with 10 text features (extracted via spaCy + VADER) to dynamically predict optimal MCTS search width (2–9) for each question, replacing fixed-width strategies.
- **Risk-Adaptive Search**: Employs Bayesian risk minimization to dynamically evaluate reasoning states and optimize search strategies.
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
│   ├── demo.json                        # Sample dataset (MuSiQue format)
│   ├── train.csv                        # Training set (10 text features + labels)
│   └── test.csv                         # Test set (10 text features + labels)
├── train_models/
│   ├── train_models.py                  # Train RF/XGB/LDA width predictors
│   ├── extract_10_features.py           # Feature extraction utilities
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

## Adaptive Width Predictor

The width predictor uses a trained **LDA (Linear Discriminant Analysis)** model to predict the optimal MCTS search width (2–9) for each input question. It extracts 10 text features using spaCy and VADER sentiment analysis:

| Feature | Description |
|---------|-------------|
| `length_ratio` | Word count of the question |
| `num_entities` | Number of named entities |
| `sentence_complexity` | Count of complex dependency relations (conj, relcl, ccomp, xcomp) |
| `keyword_density` | Ratio of content words (NOUN, ADJ, PROPN) |
| `sentiment_strength` | Absolute compound sentiment score |
| `dependency_depth` | Maximum depth of dependency parse tree |
| `multi_entity_flag` | Binary flag: 1 if entities > 2 |
| `pronoun_density` | Ratio of pronouns in text |
| `verb_density` | Ratio of verbs in text |
| `question_word_flag` | Binary flag for question word type |

**Performance** (on test set):
| Model | Strict Accuracy | Relaxed Accuracy |
|-------|----------------|-----------------|
| Random Forest | 43.96% | 73.27% |
| XGBoost | 35.59% | 70.85% |
| **LDA** | **44.77%** | **73.59%** |

> *Relaxed Accuracy*: prediction is correct if the predicted width is in the set of widths that can produce a correct answer for that question.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

Edit `config.json` with your API credentials:

```json
{
  "KIMI_API_KEY": "your-kimi-api-key",
  "KIMI_BASE_URL": "your-url",
  "HUGGINGFACEHUB_API_TOKEN": "your-hf-token"
}
```

### 3. Run Main Experiment

```bash
python evaluate_width_with_question.py
```

This will:
1. Load the trained LDA width predictor
2. For each question in `data/demo.json`, predict the optimal width
3. Run MCTS with the predicted width
4. Output exact match accuracy and F1 scores

### 4. Run Comparison Experiment

```bash
python inference_analysis/compare_fixed_vs_adaptive_musique.py
```

Compares fixed-width MCTS (width=2,3,...,9) against adaptive-width (LDA-predicted) on the MuSiQue dataset.

## Training the Width Predictor

To retrain the width prediction models:

```bash
cd train_models
python train_models.py
```

This trains three models (Random Forest, XGBoost, LDA) on the 10 text features and evaluates with both strict and relaxed accuracy. The best model (LDA) is saved to `relaxed_acc_results/lda_width_predictor.pkl`.

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

## Evaluation Metrics

- **Exact Match (EM)**: Whether the predicted answer exactly matches the gold answer
- **F1 Score**: Token-level F1 between predicted and gold answers
- **Strict Accuracy** (for width predictor): predicted width == best width
- **Relaxed Accuracy** (for width predictor): predicted width ∈ correct widths set
