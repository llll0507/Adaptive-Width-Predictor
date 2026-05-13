#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import logging
import sys

# Load API Key first (set environment variables before importing src.task)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # Project root is the script directory itself
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

if not os.path.exists(CONFIG_PATH):
    print(f"Error: Configuration file not found {CONFIG_PATH}")
    sys.exit(1)

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    if 'KIMI_API_KEY' not in config:
        sys.exit(1)
    os.environ["API_KEY"] = config['KIMI_API_KEY']
    if 'KIMI_BASE_URL' in config:
        os.environ["BASE_URL"] = config['KIMI_BASE_URL']
except Exception as e:
    print(f"Loading configuration failed: {e}")
    sys.exit(1)

# Now import project modules
import numpy as np
import pickle
import spacy
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.task import MCTSTask
from utils.verify import exact_match, F1_score_compute
from utils.inference_model import cleanup_model, initialize_model

# Experiment configuration
SAMPLE_SIZE = 3
SEED = 42
FIXED_DEPTH = 4
DATA_PATH = os.path.join(PROJECT_ROOT, "data/demo.json")  # demo dataset

# LDA Width Predictor checkpoint path
LDA_CHECKPOINT = Path(
    os.path.join(PROJECT_ROOT, "")
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Question words for feature extraction
WHAT_WORDS = {"what", "which", "who", "whom", "whose"}
HOW_WORDS = {"how", "how many", "how much", "how long", "how often"}
WHY_WORDS = {"why", "what for", "for what reason"}

# Load NLP models once at module level
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()


# --- LDA Width Predictor ---
class WidthPredictor:
    """Predict optimal MCTS width (2-9) using trained LDA model."""
    
    def __init__(self):
        print("Initializing prediction engine (LDA)...")
        assert LDA_CHECKPOINT.exists(), f"Checkpoint not found: {LDA_CHECKPOINT}"
        
        with open(LDA_CHECKPOINT, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.features = checkpoint['features']
        self.results = checkpoint['results']
        
        print(
            f"LDA checkpoint loaded "
            f"(strict_acc={self.results['strict_acc']*100:.2f}%, "
            f"relaxed_acc={self.results['relaxed_acc']*100:.2f}%)"
        )
    
    def _extract_features(self, text: str) -> dict:
        """Extract 10 text features from question."""
        doc = nlp(text.lower())
        
        # 1. length_ratio
        length_ratio = len(text.split())
        
        # 2. num_entities
        num_entities = len(doc.ents)
        
        # 3. sentence_complexity
        sentence_complexity = sum(
            1 for token in doc if token.dep_ in {"conj", "relcl", "ccomp", "xcomp"}
        )
        
        # 4. keyword_density
        content_words = sum(
            1 for token in doc if token.pos_ in {"NOUN", "ADJ", "PROPN"}
        )
        keyword_density = content_words / max(len(doc), 1)
        
        # 5. sentiment_strength
        sentiment_strength = abs(analyzer.polarity_scores(text)['compound'])
        
        # 6. dependency_depth
        def get_depth(token, visited=None):
            if visited is None:
                visited = set()
            if token.i in visited:
                return 0
            visited.add(token.i)
            if not list(token.children):
                return 1
            return 1 + max(get_depth(child, visited) for child in token.children)
        
        dependency_depth = max((get_depth(token) for token in doc), default=0)
        
        # 7. multi_entity_flag
        multi_entity_flag = 1 if num_entities > 2 else 0
        
        # 8. pronoun_density
        pronoun_density = sum(
            1 for token in doc if token.pos_ == "PRON"
        ) / max(len(doc), 1)
        
        # 9. verb_density
        verb_density = sum(
            1 for token in doc if token.pos_ == "VERB"
        ) / max(len(doc), 1)
        
        # 10. question_word_flag
        words = set(text.lower().split())
        has_what = 1 if words & WHAT_WORDS else 0
        has_how = 1 if any(h in text.lower() for h in HOW_WORDS) else 0
        has_why = 1 if any(w in text.lower() for w in WHY_WORDS) else 0
        question_word_flag = max(has_what, has_how, has_why)
        
        return {
            "length_ratio": length_ratio,
            "num_entities": num_entities,
            "sentence_complexity": sentence_complexity,
            "keyword_density": keyword_density,
            "sentiment_strength": sentiment_strength,
            "dependency_depth": dependency_depth,
            "multi_entity_flag": multi_entity_flag,
            "pronoun_density": pronoun_density,
            "verb_density": verb_density,
            "question_word_flag": question_word_flag,
        }
    
    def predict(self, question: str) -> int:
        """Predict width for a single question."""
        features = self._extract_features(question)
        feature_vector = np.array([[features[feat] for feat in self.features]])
        prediction = int(self.model.predict(feature_vector)[0])
        return prediction

# --- Helper Functions ---
def setup_logging():
    log_file = f"adaptive_width_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()], force=True)
    return log_file

def compute_retrieval_f1(data, facts_text):
    gold_titles = {sf[0] for sf in data.get("supporting_facts", []) if sf}
    if not gold_titles: return 0.0
    pred_titles = set()
    if facts_text:
        for line in facts_text.splitlines():
            line = line.strip()
            if not line: continue
            title = line.split(":", 1)[0].strip() if ":" in line else line
            if title: pred_titles.add(title)
    if not pred_titles: return 0.0
    inter = gold_titles & pred_titles
    p, r = len(inter)/len(pred_titles), len(inter)/len(gold_titles)
    return 2*p*r/(p+r) if p+r > 0 else 0.0

def main():
    print("="*80 + "\nAdaptive Width MCTS Experiment\n" + "="*80)
    setup_logging()
    
    # 1. Initialize prediction engine (RoBERTa-base)
    predictor = WidthPredictor()
    
    # 2. Load dataset
    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 3. Random sample selection
    import random
    random.seed(SEED)
    selected_indices = random.sample(range(len(dataset)), SAMPLE_SIZE)
    print(f"Randomly sampled {SAMPLE_SIZE} questions (SEED={SEED}), indices: {selected_indices})")
    print(f"Total {len(selected_indices)} questions.")

    # 4. Experiment loop
    results = []
    initialize_model()
    
    for i, idx in enumerate(tqdm(selected_indices, desc="Adaptive MCTS")):
        data = dataset[idx]
        question = data["question"]
        
        # --- Core: predict width ---
        pred_width = predictor.predict(question)
        logging.info(f"Sample {idx} | Predicted width: {pred_width} | Question: {question[:50]}...")
        
        task = MCTSTask(
            time_limit=None, iteration_limit=30, exploration_constant=1.4,
            multihops=int(pred_width), total_depth=FIXED_DEPTH,
            data=data, data_path=DATA_PATH, data_idx=idx,
            alpha=0.1, temperature=0.7, ans_weight=1, max_tokens=2048,
            seed=170, do_sample=True, max_new_tokens=1024,
            run_mode="MCTS", value_mode="risk",
            value_model=""
        )
        
        try:
            root_node = task.run()
            best_leaf = task.get_best_path()
            is_correct = exact_match(best_leaf["answer"], data["answer"])
            ret_f1 = compute_retrieval_f1(data, best_leaf["facts"])
            ans_f1 = F1_score_compute(data["answer"], best_leaf["answer"])
            
            results.append({
                "idx": int(idx), "pred_width": int(pred_width),
                "correct": is_correct, "retrieval_f1": ret_f1, "answer_f1": ans_f1
            })
            
        except Exception as e:
            logging.error(f"Failed to process sample {idx}: {e}")

    # 5. Save results
    # Ensure output directory exists
    output_dir = os.path.join(SCRIPT_DIR, "Results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON file to results directory (with musique identifier)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"musique_adaptive_results_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # 6. Real-time statistical analysis
    total = len(results)
    if total == 0:
        print("\nError: No samples were successfully processed.")
        return
    
    correct_count = sum(1 for r in results if r["correct"])
    acc = correct_count / total
    avg_ret_f1 = np.mean([r["retrieval_f1"] for r in results])
    avg_ans_f1 = np.mean([r["answer_f1"] for r in results])
    
    # Count questions and correct answers for each width
    from collections import Counter
    width_counts = Counter(r["pred_width"] for r in results)
    width_correct = Counter(r["pred_width"] for r in results if r["correct"])
    
    print("\n" + "="*80)
    print("🎯 Adaptive Width MCTS Experiment Results Summary")
    print("="*80)
    print(f"Total samples: {total}")
    print(f"\n[Core Performance Metrics]")
    print(f"  Accuracy (EM Accuracy):        {correct_count} / {total} = {acc:.2%}")
    print(f"  Avg Retrieval F1:              {avg_ret_f1:.4f}")
    print(f"  Avg Answer F1:                 {avg_ans_f1:.4f}")
    
    print(f"\n[Predicted Width Distribution Statistics]")
    print(f"{'Width':<8} {'Count':<10} {'Ratio':<13} {'Accuracy (Correct/Total)':<30}")
    print("-" * 80)
    for width in sorted(width_counts.keys()):
        count = width_counts[width]
        correct = width_correct.get(width, 0)
        w_acc = correct / count if count > 0 else 0
        percentage = count / total * 100
        print(f"{width:<8} {count:<10} {percentage:<10.2f} % {w_acc:.2%}     ({correct}/{count})")
    
    print(f"\n[Predicted Width for Each Question]")
    print(f"{'Question Index (idx)':<20} {'Predicted Width':<12} {'Correct':<10}")
    print("-" * 50)
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"{r['idx']:<20} {r['pred_width']:<12} {status:<10}")
    
    # Generate statistical summary text file
    summary_file = os.path.join(output_dir, f"musique_adaptive_results_{timestamp}_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("🎯 Adaptive Width MCTS Experiment Results Summary (MuSiQue Dataset)\n")
        f.write("="*80 + "\n")
        f.write(f"Total samples: {total}\n\n")
        f.write("[Core Performance Metrics]\n")
        f.write(f"  Accuracy (EM Accuracy):        {correct_count} / {total} = {acc:.2%}\n")
        f.write(f"  Avg Retrieval F1:              {avg_ret_f1:.4f}\n")
        f.write(f"  Avg Answer F1:                 {avg_ans_f1:.4f}\n\n")
        
        f.write("[Predicted Width Distribution Statistics]\n")
        f.write(f"{'Width':<8} {'Count':<10} {'Ratio':<13} {'Accuracy (Correct/Total)':<30}\n")
        f.write("-" * 80 + "\n")
        for width in sorted(width_counts.keys()):
            count = width_counts[width]
            correct = width_correct.get(width, 0)
            w_acc = correct / count if count > 0 else 0
            percentage = count / total * 100
            f.write(f"{width:<8} {count:<10} {percentage:<10.2f} % {w_acc:.2%}     ({correct}/{count})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Detailed results saved to: {output_file}\n")
        f.write("="*80 + "\n")
    
    print("\n" + "="*80)
    print(f"Detailed results saved to: {output_file}")
    print(f"Statistical summary saved to: {summary_file}")
    print("="*80)

if __name__ == "__main__":
    main()
