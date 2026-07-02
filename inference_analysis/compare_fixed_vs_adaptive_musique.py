#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# 0. Path and Environment Initialization (must be before importing src/utils)
# ============================================================
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent  # AW-MCTS-main
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config.json"
if CONFIG_PATH.exists():
    import json as _json
    with open(CONFIG_PATH, encoding="utf-8") as _f:
        _cfg = _json.load(_f)
    if "API_KEY" in _cfg:
        os.environ["API_KEY"] = _cfg["API_KEY"]

# ============================================================
# 1. Standard Library & Third-party Libraries
# ============================================================
import csv
import json
import logging
import pickle
import random
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# matplotlib: no GUI dependency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# 2. Experiment Hyperparameters (shared by all methods)
# ============================================================
SEED           = 42
N_SAMPLES      = 50
FIXED_WIDTHS   = list(range(2, 10))      # [2, 3, 4, 5, 6, 7, 8, 9]

# MCTS Parameters (identical for fixed & adaptive width)
MCTS_ITERATIONS     = 30
MCTS_DEPTH          = 4
MCTS_EXPLORATION    = 1.4
MCTS_ALPHA          = 0.1
MCTS_TEMPERATURE    = 0.7
MCTS_ANS_WEIGHT     = 1
MCTS_MAX_TOKENS     = 2048
MCTS_SEED           = 170
MCTS_DO_SAMPLE      = True
MCTS_MAX_NEW_TOKENS = 1024
VALUE_MODE          = "risk"
VALUE_MODEL         = "/media/m811/1.6T/m811/models/Qwen2.5-7B-Instruct"

DATA_PATH = PROJECT_ROOT / "data" / "demo.json"

RESULTS_DIR = Path("")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. Logging
# ============================================================
log_path = RESULTS_DIR / "experiment.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 4. Token Counting (use tiktoken, fallback to split estimation)
# ============================================================
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text)) if text else 0
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text.split()) if text else 0


# ============================================================
# 5. Monkey-patch: token counting & node expansion counting
# ============================================================
_token_store = {"input": 0, "output": 0}
_node_store   = {"visited": 0, "expanded": 0}
_llm_calls    = {"count": 0}


def reset_counters():
    _token_store["input"]  = 0
    _token_store["output"] = 0
    _node_store["visited"]  = 0
    _node_store["expanded"] = 0
    _llm_calls["count"]    = 0


def patch_token_counter():

    try:
        import utils.inference_model as im
        orig = im.call_api_for_subq_and_answer

        def _wrapped(query, **kwargs):
            # Count input tokens
            if isinstance(query, str):
                _token_store["input"] += count_tokens(query)
            elif isinstance(query, list):
                _token_store["input"] += sum(
                    count_tokens(m.get("content", "")) for m in query
                )
            _llm_calls["count"] += 1

            result = orig(query, **kwargs)

            # Count output tokens
            if result:
                for item in result:
                    if isinstance(item, dict):
                        for msg in item.get("generated_text", []):
                            _token_store["output"] += count_tokens(
                                msg.get("content", "")
                            )
            return result

        im.call__api_for_subq_and_answer = _wrapped
        logger.info("Token counter patch applied.")
    except Exception as e:
        logger.warning(f"Token patch failed: {e}")


def patch_node_counter():
    """Patch src.mcts.expand_node to count expanded nodes;
    Patch src.mcts.select_node to count visited nodes."""
    try:
        import src.mcts as mcts_mod
        orig_expand = mcts_mod.expand_node
        orig_select = mcts_mod.select_node

        def _patched_expand(node, task):
            before = len(node.children)   # Record children count before expansion
            result = orig_expand(node, task)
            # expand_node modifies node in-place and returns the same object, compare result and before
            after  = len(result.children)
            added  = max(0, after - before)
            _node_store["expanded"] += added
            return result

        def _patched_select(root, task):
            # Count nodes traversed by select: track internal traversal path length
            import src.mcts as _m
            depth = 0
            cur = root
            while cur.is_fully_expanded and cur.children:
                cur = _m.get_best_child(cur, task)
                depth += 1
            _node_store["visited"] += depth + 1  
            return orig_select(root, task)

        mcts_mod.expand_node = _patched_expand
        mcts_mod.select_node = _patched_select
        logger.info("Node counter patch applied.")
    except Exception as e:
        logger.warning(f"Node patch failed: {e}")


# Question words for feature extraction
WHAT_WORDS = {"what", "which", "who", "whom", "whose"}
HOW_WORDS = {"how", "how many", "how much", "how long", "how often"}
WHY_WORDS = {"why", "what for", "for what reason"}

# Load NLP models once at module level
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()


# ============================================================
# 6. LDA Width Predictor
# ============================================================
class LDAWidthPredictor:
    """Predict optimal MCTS width (2-9) for a question using trained LDA model."""
    
    # Trained checkpoint path
    CHECKPOINT_PATH = PROJECT_ROOT / ""

    def __init__(self):
        if not self.CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"LDA checkpoint not found: {self.CHECKPOINT_PATH}")
        
        with open(self.CHECKPOINT_PATH, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.features = checkpoint['features']
        self.results = checkpoint['results']
        
        logger.info(
            f"LDA checkpoint loaded. "
            f"strict_acc={self.results['strict_acc']:.4f}, "
            f"relaxed_acc={self.results['relaxed_acc']:.4f}"
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

    def predict_batch(self, questions: list) -> list:
        return [self.predict(q) for q in questions]


# ============================================================
# 7. Helper Functions
# ============================================================
def compute_retrieval_f1(data: dict, retrieved_titles: set) -> float:
    """
    Retrieval F1 based on document title sets.
    retrieved_titles: Set of document keywords returned by all retrieve() calls during MCTS
    gold_titles:      Set of titles from data['supporting_facts']
    """
    gold_titles = {sf[0] for sf in data.get("supporting_facts", []) if sf}
    if not gold_titles:
        return 0.0
    if not retrieved_titles:
        return 0.0
    inter = gold_titles & retrieved_titles
    if not inter:
        return 0.0
    p = len(inter) / len(retrieved_titles)
    r = len(inter) / len(gold_titles)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _extract_best_answer(root_node) -> tuple:
    """DFS traversal to find leaf node with highest value, returns (answer, node)."""
    best_val    = -float("inf")
    best_answer = ""
    stack = [root_node]
    while stack:
        node = stack.pop()
        if not node.children:
            if node.value > best_val:
                best_val    = node.value
                best_answer = node.answer or node.query or ""
        else:
            stack.extend(node.children.values())
    return best_answer


# ============================================================
# 8. Core: Run Single MCTS (with given width)
# ============================================================
def run_single_mcts(
    data: dict,
    width: int,
    data_path: str,
    data_idx: int,
) -> dict:
    """
    Run MCTS once with given width, returns dict containing all record fields.
    All parameters except width use unified constants from top of file.
    """
    from src.task import MCTSTask
    from src.mcts import MCTS_search
    from utils.verify import exact_match, F1_score_compute

    reset_counters()
    t_start = time.time()

    task = MCTSTask(
        time_limit=None,
        iteration_limit=MCTS_ITERATIONS,
        exploration_constant=MCTS_EXPLORATION,
        multihops=int(width),               # ← Only difference
        total_depth=MCTS_DEPTH,
        data=data,
        data_path=str(data_path),
        data_idx=data_idx,
        alpha=MCTS_ALPHA,
        temperature=MCTS_TEMPERATURE,
        ans_weight=MCTS_ANS_WEIGHT,
        max_tokens=MCTS_MAX_TOKENS,
        seed=MCTS_SEED,
        do_sample=MCTS_DO_SAMPLE,
        max_new_tokens=MCTS_MAX_NEW_TOKENS,
        run_mode="MCTS",
        value_mode=VALUE_MODE,
        value_model=VALUE_MODEL,
    )

    error_msg = None
    em = 0
    retrieval_f1 = 0.0
    answer = ""
    retrieved_titles: set = set()   # Collect all retrieved document keywords during MCTS

    # ── monkey-patch retrieve to record document title from each retrieval ──────────
    import utils.rag as _rag_module
    from utils.rag import get_retriever as _get_retriever

    _orig_retrieve = _rag_module.retrieve

    def _patched_retrieve(query, dp, di, topk=5):
        result = _orig_retrieve(query, dp, di, topk=topk)
        # Extract title from returned "title: facts\n..."
        if result:
            for line in result.splitlines():
                line = line.strip()
                if not line:
                    continue
                title = line.split(":", 1)[0].strip() if ":" in line else line
                if title:
                    retrieved_titles.add(title)
        return result

    _rag_module.retrieve = _patched_retrieve
    # Synchronously replace retrieve reference already imported in wrap.py
    try:
        import utils.wrap as _wrap_module
        _wrap_module.retrieve = _patched_retrieve
    except Exception:
        pass

    try:
        root_node, _ = MCTS_search(task)

        # Get final answer
        try:
            best_path = task.get_best_path()
            if best_path:
                answer = best_path.get("answer", "") or ""
        except Exception:
            answer = _extract_best_answer(root_node)

        gold = data.get("answer", "")
        em   = 1 if exact_match(answer, gold) else 0
        retrieval_f1 = compute_retrieval_f1(data, retrieved_titles)

    except Exception as exc:
        error_msg = str(exc)
        logger.warning(
            f"  [MCTS ERROR] width={width} idx={data_idx}: {error_msg[:120]}"
        )
    finally:
        # Restore original retrieve
        _rag_module.retrieve = _orig_retrieve
        try:
            _wrap_module.retrieve = _orig_retrieve
        except Exception:
            pass

    t_end = time.time()
    return {
        "em":             em,
        "retrieval_f1":   retrieval_f1,
        "input_tokens":   _token_store["input"],
        "output_tokens":  _token_store["output"],
        "total_tokens":   _token_store["input"] + _token_store["output"],
        "visited_nodes":  _node_store["visited"],
        "expanded_nodes": _node_store["expanded"],
        "llm_calls":      _llm_calls["count"],
        "inference_time_sec": round(t_end - t_start, 2),
        "answer":         answer,
        "error":          error_msg,
    }


# ============================================================
# 9. Save & Plot
# ============================================================
PER_Q_FIELDS = [
    "question_id", "question", "method", "width",
    "em", "retrieval_f1",
    "input_tokens", "output_tokens", "total_tokens",
    "visited_nodes", "expanded_nodes", "llm_calls",
    "inference_time_sec", "error",
]

SUMMARY_FIELDS = [
    "method", "width_label",
    "avg_em", "avg_retrieval_f1",
    "avg_input_tokens", "avg_output_tokens", "avg_total_tokens",
    "avg_visited_nodes", "avg_expanded_nodes",
    "avg_llm_calls", "avg_inference_time_sec",
    "n_success",
]


def save_per_question(rows: list):
    path = RESULTS_DIR / "per_question_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_Q_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"per_question_results saved → {path}")


def build_summary(rows: list) -> list:
    """Aggregate by method, returns list[dict]."""
    grouped: dict[str, list] = defaultdict(list)
    for r in rows:
        grouped[r["method"]].append(r)

    summary_rows = []
    # Fixed width: sort by width
    for w in FIXED_WIDTHS:
        method = f"fixed_width_{w}"
        grp    = grouped.get(method, [])
        if not grp:
            continue
        row = _agg(method, str(w), grp)
        summary_rows.append(row)
    # Adaptive width
    method = "adaptive_lda"
    grp    = grouped.get(method, [])
    if grp:
        summary_rows.append(_agg(method, "adaptive", grp))
    return summary_rows


def _agg(method: str, width_label: str, grp: list) -> dict:
    valid = [r for r in grp if r.get("error") is None]
    n = len(valid)
    if n == 0:
        return {f: 0.0 for f in SUMMARY_FIELDS} | {"method": method, "width_label": width_label, "n_success": 0}
    return {
        "method":              method,
        "width_label":         width_label,
        "avg_em":              round(np.mean([r["em"] for r in valid]), 4),
        "avg_retrieval_f1":    round(np.mean([r["retrieval_f1"] for r in valid]), 4),
        "avg_input_tokens":    round(np.mean([r["input_tokens"] for r in valid]), 2),
        "avg_output_tokens":   round(np.mean([r["output_tokens"] for r in valid]), 2),
        "avg_total_tokens":    round(np.mean([r["total_tokens"] for r in valid]), 2),
        "avg_visited_nodes":   round(np.mean([r["visited_nodes"] for r in valid]), 2),
        "avg_expanded_nodes":  round(np.mean([r["expanded_nodes"] for r in valid]), 2),
        "avg_llm_calls":       round(np.mean([r["llm_calls"] for r in valid]), 2),
        "avg_inference_time_sec": round(np.mean([r["inference_time_sec"] for r in valid]), 2),
        "n_success":           n,
    }


def save_summary(summary_rows: list):
    path = RESULTS_DIR / "summary_results.csv"
    # Construct fixed width average row
    fixed_rows = [r for r in summary_rows if r["method"].startswith("fixed_width_")]
    rows_to_write = list(summary_rows)
    if fixed_rows:
        def _mean(key):
            return sum(r[key] for r in fixed_rows) / len(fixed_rows)
        avg_row = {field: "" for field in SUMMARY_FIELDS}
        avg_row["method"]                 = "fixed_width_avg"
        avg_row["width_label"]            = "avg"
        avg_row["avg_em"]                 = round(_mean("avg_em"), 4)
        avg_row["avg_retrieval_f1"]       = round(_mean("avg_retrieval_f1"), 4)
        avg_row["avg_total_tokens"]       = round(_mean("avg_total_tokens"), 2)
        avg_row["avg_expanded_nodes"]     = round(_mean("avg_expanded_nodes"), 2)
        avg_row["avg_inference_time_sec"] = round(_mean("avg_inference_time_sec"), 2)
        avg_row["n_success"]              = round(_mean("n_success"), 2)
        rows_to_write.append(avg_row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_to_write)
    logger.info(f"summary_results saved → {path}")


def plot_results(summary_rows: list, adaptive_width_dist: Counter):
    """Generate four plots."""
    fixed = [r for r in summary_rows if r["method"].startswith("fixed_width_")]
    fixed.sort(key=lambda r: int(r["width_label"]))
    adaptive_rows = [r for r in summary_rows if r["method"] == "adaptive_lda"]
    adaptive = adaptive_rows[0] if adaptive_rows else None

    widths = [int(r["width_label"]) for r in fixed]
    em_vals     = [r["avg_em"] for r in fixed]
    token_vals  = [r["avg_total_tokens"] for r in fixed]
    nodes_vals  = [r["avg_expanded_nodes"] for r in fixed]

    # ── Plot 1: EM vs Width ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(widths, em_vals, marker="o", linewidth=2, label="Fixed Width")
    if adaptive:
        ax.axhline(adaptive["avg_em"], color="red", linestyle="--", linewidth=1.5,
                   label=f"Adaptive (LDA) EM={adaptive['avg_em']:.3f}")
    ax.set_xlabel("Fixed Search Width")
    ax.set_ylabel("Exact Match (EM)")
    ax.set_title("EM vs. Fixed Width on MuSiQue")
    ax.set_xticks(widths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "plot_em_vs_width.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: avg total tokens vs Width ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(widths, token_vals, marker="s", color="steelblue", linewidth=2,
            label="Fixed Width")
    if adaptive:
        ax.axhline(adaptive["avg_total_tokens"], color="red", linestyle="--",
                   linewidth=1.5,
                   label=f"Adaptive (LDA) tokens={adaptive['avg_total_tokens']:.0f}")
    ax.set_xlabel("Fixed Search Width")
    ax.set_ylabel("Avg Total Tokens")
    ax.set_title("Avg Total Tokens vs. Fixed Width on MuSiQue")
    ax.set_xticks(widths)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "plot_tokens_vs_width.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: avg expanded nodes vs Width ───────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(widths, nodes_vals, marker="^", color="darkorange", linewidth=2,
            label="Fixed Width")
    if adaptive:
        ax.axhline(adaptive["avg_expanded_nodes"], color="red", linestyle="--",
                   linewidth=1.5,
                   label=f"Adaptive (LDA) nodes={adaptive['avg_expanded_nodes']:.1f}")
    ax.set_xlabel("Fixed Search Width")
    ax.set_ylabel("Avg Expanded Nodes")
    ax.set_title("Avg Expanded Nodes vs. Fixed Width on MuSiQue")
    ax.set_xticks(widths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "plot_nodes_vs_width.png", dpi=150)
    plt.close(fig)

    # ── Plot 4: Adaptive Width Prediction Distribution ──────────────────────────────────
    if adaptive_width_dist:
        ws     = sorted(adaptive_width_dist.keys())
        counts = [adaptive_width_dist[w] for w in ws]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(w) for w in ws], counts, color="teal", edgecolor="black")
        ax.set_xlabel("Predicted Width")
        ax.set_ylabel("Count")
        ax.set_title("LDA Predicted Width Distribution (N=100)")
        for x, y in zip([str(w) for w in ws], counts):
            ax.text(x, y + 0.3, str(y), ha="center", fontsize=9)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "plot_adaptive_width_hist.png", dpi=150)
        plt.close(fig)

    logger.info("All plots saved.")


# ============================================================
# 10. Markdown Experiment Summary
# ============================================================
def write_markdown_summary(summary_rows: list, adaptive_width_dist: Counter,
                           sampled_count: int):
    fixed = [r for r in summary_rows if r["method"].startswith("fixed_width_")]
    fixed.sort(key=lambda r: int(r["width_label"]))
    adaptive_rows = [r for r in summary_rows if r["method"] == "adaptive_lda"]
    adaptive = adaptive_rows[0] if adaptive_rows else None

    lines = []
    lines.append("# MuSiQue Fixed Width vs Adaptive Width Experiment Summary\n")
    lines.append(f"- **Dataset**: MuSiQue (`musique_unified.json`)")
    lines.append(f"- **Sample Count**: {sampled_count} (random seed = {SEED})")
    lines.append(f"- **MCTS Iterations**: {MCTS_ITERATIONS}")
    lines.append(f"- **MCTS Depth**: {MCTS_DEPTH}")
    lines.append(f"- **Value Function Mode**: {VALUE_MODE}")
    lines.append(f"- **Token Counting**: tiktoken `cl100k_base` estimation")
    lines.append("")

    # Fixed width summary table
    lines.append("## Fixed Width Method Summary\n")
    lines.append("| width | avg EM | avg Ret-F1 | avg Total Tokens | avg Expanded Nodes | avg LLM Calls | avg Time (s) |")
    lines.append("|------|--------|------------|------------------|--------------------|---------------|--------------|")
    for r in fixed:
        lines.append(
            f"| {r['width_label']} "
            f"| {r['avg_em']:.4f} "
            f"| {r['avg_retrieval_f1']:.4f} "
            f"| {r['avg_total_tokens']:.0f} "
            f"| {r['avg_expanded_nodes']:.1f} "
            f"| {r['avg_llm_calls']:.1f} "
            f"| {r['avg_inference_time_sec']:.1f} |"
        )
    lines.append("")

    # Adaptive width summary
    if adaptive:
        lines.append("## Adaptive Width (LDA) Summary\n")
        lines.append("| method | avg EM | avg Ret-F1 | avg Total Tokens | avg Expanded Nodes | avg LLM Calls | avg Time (s) |")
        lines.append("|------|--------|------------|------------------|--------------------|---------------|--------------|")
        lines.append(
            f"| adaptive_lda "
            f"| {adaptive['avg_em']:.4f} "
            f"| {adaptive['avg_retrieval_f1']:.4f} "
            f"| {adaptive['avg_total_tokens']:.0f} "
            f"| {adaptive['avg_expanded_nodes']:.1f} "
            f"| {adaptive['avg_llm_calls']:.1f} "
            f"| {adaptive['avg_inference_time_sec']:.1f} |"
        )
        lines.append("")

        # Compare with best fixed width
        best_fixed = max(fixed, key=lambda r: r["avg_em"])
        lines.append("## Adaptive Width vs Best Fixed Width Comparison\n")
        lines.append(f"Best Fixed Width: **{best_fixed['width_label']}** (avg EM = {best_fixed['avg_em']:.4f})\n")
        em_delta     = adaptive["avg_em"] - best_fixed["avg_em"]
        token_delta  = adaptive["avg_total_tokens"] - best_fixed["avg_total_tokens"]
        token_pct    = token_delta / best_fixed["avg_total_tokens"] * 100 if best_fixed["avg_total_tokens"] > 0 else float("nan")
        lines.append(f"- EM Change: {em_delta:+.4f}")
        lines.append(f"- avg Total Tokens Change: {token_delta:+.0f} ({token_pct:+.1f}%)")
        lines.append("")

    # Adaptive width distribution
    if adaptive_width_dist:
        lines.append("## Adaptive Width Prediction Distribution\n")
        lines.append("| Predicted Width | Count | Ratio |")
        lines.append("|-----------------|-------|-------|")
        total_adp = sum(adaptive_width_dist.values())
        for w in sorted(adaptive_width_dist.keys()):
            cnt = adaptive_width_dist[w]
            lines.append(f"| {w} | {cnt} | {cnt/total_adp*100:.1f}% |")
        lines.append("")

    lines.append("## Output Files\n")
    lines.append(f"- `{RESULTS_DIR}/sampled_questions.json`")
    lines.append(f"- `{RESULTS_DIR}/per_question_results.csv`")
    lines.append(f"- `{RESULTS_DIR}/summary_results.csv`")
    lines.append(f"- `{RESULTS_DIR}/plot_em_vs_width.png`")
    lines.append(f"- `{RESULTS_DIR}/plot_tokens_vs_width.png`")
    lines.append(f"- `{RESULTS_DIR}/plot_nodes_vs_width.png`")
    lines.append(f"- `{RESULTS_DIR}/plot_adaptive_width_hist.png`")
    lines.append("")
    lines.append(f"*Experiment End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    md_path = RESULTS_DIR / "experiment_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown summary saved → {md_path}")


# ============================================================
# 11. Main Process
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("MuSiQue: Fixed Width 2-9 vs LDA Adaptive Width Comparison Experiment")
    logger.info("=" * 70)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Dataset Path: {DATA_PATH}")
    logger.info(f"Results Dir:  {RESULTS_DIR}")

    # ── Load & Initialize ──────────────────────────────────────────
    try:
        from utils.inference_model import initialize_model
        if not initialize_model():
            raise RuntimeError("MCTS inference model initialization failed")
    except Exception as e:
        logger.error(f"Inference model initialization failed: {e}")
        return

    patch_token_counter()
    patch_node_counter()

    # ── Load Dataset ─────────────────────────────────────────────
    logger.info(f"Loading dataset: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        full_dataset = json.load(f)
    logger.info(f"Dataset has {len(full_dataset)} items")

    # ── Fix random seed, sample 100 questions (shared by all methods) ──────────────
    rng = random.Random(SEED)
    indices = list(range(len(full_dataset)))
    rng.shuffle(indices)
    selected_indices = indices[:N_SAMPLES]
    sampled_items    = [full_dataset[i] for i in selected_indices]

    # Save sampled questions
    sampled_save = [
        {"dataset_index": int(idx), "question": item.get("question", ""),
         "answer": item.get("answer", "")}
        for idx, item in zip(selected_indices, sampled_items)
    ]
    with open(RESULTS_DIR / "sampled_questions.json", "w", encoding="utf-8") as f:
        json.dump(sampled_save, f, indent=2, ensure_ascii=False)
    logger.info(f"Sampling complete, {len(sampled_items)} questions (seed={SEED})")

    # ── Load LDA predictor, batch predict widths ─────────────────────
    logger.info("Loading LDA width predictor ...")
    predictor = LDAWidthPredictor()

    questions_list = [item.get("question", "") for item in sampled_items]
    logger.info("Batch predicting adaptive widths ...")
    pred_widths = predictor.predict_batch(questions_list)
    adaptive_width_dist = Counter(pred_widths)
    logger.info(f"Adaptive width distribution: {dict(sorted(adaptive_width_dist.items()))}")

    # ── Main Experiment Loop ──────────────────────────────────────────────
    per_question_rows: list[dict] = []

    # ---- Fixed width: width ∈ [2, 9] ----
    for w in FIXED_WIDTHS:
        method = f"fixed_width_{w}"
        logger.info(f"\n{'='*50}")
        logger.info(f"Method: {method}")
        logger.info(f"{'='*50}")

        for i, (item, idx) in enumerate(zip(sampled_items, selected_indices)):
            qid      = item.get("_id", str(idx))
            question = item.get("question", "")
            logger.info(
                f"  [{i+1}/{N_SAMPLES}] {method} | idx={idx} | {question[:60]}..."
            )
            try:
                result = run_single_mcts(
                    data=item, width=w,
                    data_path=DATA_PATH, data_idx=idx,
                )
            except Exception as exc:
                logger.error(f"  Unexpected exception {exc}\n{traceback.format_exc()[:300]}")
                result = {
                    "em": 0, "retrieval_f1": 0.0,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "visited_nodes": 0, "expanded_nodes": 0,
                    "llm_calls": 0, "inference_time_sec": 0.0,
                    "error": str(exc),
                }

            per_question_rows.append({
                "question_id":        qid,
                "question":           question,
                "method":             method,
                "width":              w,
                "em":                 result["em"],
                "retrieval_f1":       result["retrieval_f1"],
                "input_tokens":       result["input_tokens"],
                "output_tokens":      result["output_tokens"],
                "total_tokens":       result["total_tokens"],
                "visited_nodes":      result["visited_nodes"],
                "expanded_nodes":     result["expanded_nodes"],
                "llm_calls":          result["llm_calls"],
                "inference_time_sec": result["inference_time_sec"],
                "error":              result.get("error"),
            })

    # ---- Adaptive width (LDA) ----
    method = "adaptive_lda"
    logger.info(f"\n{'='*50}")
    logger.info(f"方法: {method}")
    logger.info(f"{'='*50}")

    for i, (item, idx, pw) in enumerate(
        zip(sampled_items, selected_indices, pred_widths)
    ):
        qid      = item.get("_id", str(idx))
        question = item.get("question", "")
        logger.info(
            f"  [{i+1}/{N_SAMPLES}] {method} | idx={idx} | "
            f"pred_width={pw} | {question[:50]}..."
        )
        try:
            result = run_single_mcts(
                data=item, width=pw,
                data_path=DATA_PATH, data_idx=idx,
            )
        except Exception as exc:
            logger.error(f"  Unexpected exception {exc}\n{traceback.format_exc()[:300]}")
            result = {
                "em": 0, "retrieval_f1": 0.0,
                "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "visited_nodes": 0, "expanded_nodes": 0,
                "llm_calls": 0, "inference_time_sec": 0.0,
                "error": str(exc),
            }

        per_question_rows.append({
            "question_id":        qid,
            "question":           question,
            "method":             method,
            "width":              pw,       # Record predicted width
            "em":                 result["em"],
            "retrieval_f1":       result["retrieval_f1"],
            "input_tokens":       result["input_tokens"],
            "output_tokens":      result["output_tokens"],
            "total_tokens":       result["total_tokens"],
            "visited_nodes":      result["visited_nodes"],
            "expanded_nodes":     result["expanded_nodes"],
            "llm_calls":          result["llm_calls"],
            "inference_time_sec": result["inference_time_sec"],
            "error":              result.get("error"),
        })

    # ── Save & Summarize & Plot ──────────────────────────────────────
    save_per_question(per_question_rows)
    summary_rows = build_summary(per_question_rows)
    save_summary(summary_rows)
    plot_results(summary_rows, adaptive_width_dist)
    write_markdown_summary(summary_rows, adaptive_width_dist, len(sampled_items))

    # ── Console Print Summary ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Experiment Complete! Summary Results:")
    print("=" * 70)
    header = f"{'Method':<22} {'avg EM':>8} {'avg Ret-F1':>11} {'avg Tokens':>12} {'avg Nodes':>11} {'avg Time(s)':>12}"
    print(header)
    print("-" * 70)
    for r in summary_rows:
        label = r["method"] if r["method"] != "adaptive_roberta" else "adaptive_roberta"
        print(
            f"{label:<22} "
            f"{r['avg_em']:>8.4f} "
            f"{r['avg_retrieval_f1']:>11.4f} "
            f"{r['avg_total_tokens']:>12.0f} "
            f"{r['avg_expanded_nodes']:>11.1f} "
            f"{r['avg_inference_time_sec']:>12.1f}"
        )
    # ── Fixed Width Average Row ─────────────────────────────────────────────
    fixed_rows = [r for r in summary_rows if r["method"].startswith("fixed_width_")]
    if fixed_rows:
        def _mean(key):
            return sum(r[key] for r in fixed_rows) / len(fixed_rows)
        print("-" * 70)
        print(
            f"{'fixed_width_avg':<22} "
            f"{_mean('avg_em'):>8.4f} "
            f"{_mean('avg_retrieval_f1'):>11.4f} "
            f"{_mean('avg_total_tokens'):>12.0f} "
            f"{_mean('avg_expanded_nodes'):>11.1f} "
            f"{_mean('avg_inference_time_sec'):>12.1f}"
        )
    print("=" * 70)
    print(f"\nAll results saved to: {RESULTS_DIR}")

    # Save adaptive width prediction distribution
    dist_path = RESULTS_DIR / "adaptive_width_distribution.json"
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "n_samples": N_SAMPLES,
                "width_distribution": {str(k): v for k, v in sorted(adaptive_width_dist.items())},
                "avg_predicted_width": round(
                    sum(k * v for k, v in adaptive_width_dist.items()) / sum(adaptive_width_dist.values()), 2
                ) if adaptive_width_dist else None,
            },
            f, indent=2, ensure_ascii=False
        )
    logger.info(f"Adaptive width distribution saved → {dist_path}")
    logger.info("Experiment ended.")


if __name__ == "__main__":
    main()
