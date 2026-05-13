#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Width Prediction - Relaxed Accuracy with "suitable_widths"
=====================================================
1. Load correct_widths_mapping.json (all widths where each question is answered correctly)
2. Add "suitable_widths" column to train/test
3. Train RF/XGB with best_width as label (single-label, unchanged)
4. Evaluate with both strict_acc (pred==best_width) and relaxed_acc (pred in suitable_widths)
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = SCRIPT_DIR
TRAIN_CSV = str(BASE_DIR.parent / "data" / "train.csv")
TEST_CSV = str(BASE_DIR.parent / "data" / "test.csv")
MAPPING_JSON = str(BASE_DIR / "feature_data" / "correct_widths_mapping.json")
OUTPUT_DIR = BASE_DIR / "relaxed_acc_results"

WIDTHS_ALL = [2, 3, 4, 5, 6, 7, 8, 9]

TEXT_FEATURES = [
    'length_ratio', 'num_entities', 'sentence_complexity', 'keyword_density',
    'sentiment_strength', 'dependency_depth', 'multi_entity_flag', 'pronoun_density',
    'verb_density', 'question_word_flag',
]
FEAT_11 = TEXT_FEATURES + ['w2_avg_value']


# ══════════════════════════════════════════════════════════════
#  Load mapping & add "suitable_widths" column
# ══════════════════════════════════════════════════════════════

def load_correct_widths_mapping():
    """Load {exp_folder|idx: [correct_widths]} mapping."""
    with open(MAPPING_JSON, 'r') as f:
        raw = json.load(f)
    return raw


def add_suitable_widths_column(df, mapping):
    """Add 'suitable_widths' column: list of all widths where this question is correct."""
    suitable = []
    for _, row in df.iterrows():
        key = f"{row['exp_folder']}|{row['question_idx']}"
        if key in mapping and len(mapping[key]) > 0:
            suitable.append(mapping[key])
        else:
            # Fallback: use best_width as the only correct width
            suitable.append([int(row['best_width'])])
    df['suitable_widths'] = suitable
    return df


# ══════════════════════════════════════════════════════════════
#  Evaluation (strict + relaxed)
# ══════════════════════════════════════════════════════════════

def compute_range_accuracy(y_true, y_pred, suitable_widths, mode='strict'):
    """Compute per-range accuracy. mode='strict' or 'relaxed'."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m2 = (y_true == 2)
    m34 = np.isin(y_true, [3, 4])
    m59 = np.isin(y_true, [5, 6, 7, 8, 9])

    if mode == 'strict':
        correct = (y_pred == y_true)
    else:
        correct = np.array([p in sw for p, sw in zip(y_pred, suitable_widths)])

    acc_2 = float(correct[m2].mean()) if m2.sum() > 0 else 0.0
    acc_34 = float(correct[m34].mean()) if m34.sum() > 0 else 0.0
    acc_59 = float(correct[m59].mean()) if m59.sum() > 0 else 0.0
    return acc_2, acc_34, acc_59


def evaluate(name, y_test, y_pred, suitable_widths):
    """Evaluate with both strict and relaxed accuracy."""
    y_test, y_pred = np.array(y_test), np.array(y_pred)

    # Strict: pred == best_width
    strict_acc = float((y_pred == y_test).mean())
    pm1_acc = float((np.abs(y_pred - y_test) <= 1).mean())
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    s_2, s_34, s_59 = compute_range_accuracy(y_test, y_pred, suitable_widths, 'strict')

    # Relaxed: pred in suitable_widths
    relaxed_correct = np.array([p in sw for p, sw in zip(y_pred, suitable_widths)])
    relaxed_acc = float(relaxed_correct.mean())
    r_2, r_34, r_59 = compute_range_accuracy(y_test, y_pred, suitable_widths, 'relaxed')

    print(f"  {name}:")
    print(f"    strict_acc  = {strict_acc:.4f}   relaxed_acc = {relaxed_acc:.4f}")
    print(f"    pm1_acc     = {pm1_acc:.4f}")   


    return {
        "strict_acc": strict_acc,
        "relaxed_acc": relaxed_acc,
        "pm1_acc": pm1_acc,
        "macro_f1": macro_f1,
        "strict_2": s_2, "strict_34": s_34, "strict_59": s_59,
        "relaxed_2": r_2, "relaxed_34": r_34, "relaxed_59": r_59,
    }


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  Width Prediction \u2014 Relaxed Accuracy (suitable_widths)")
    print("=" * 70)

    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_df = train_df[train_df['best_width'].isin(WIDTHS_ALL)].reset_index(drop=True)
    test_df = test_df[test_df['best_width'].isin(WIDTHS_ALL)].reset_index(drop=True)

    # Parse correct_widths column: convert string representation to list
    import ast
    train_df['suitable_widths'] = train_df['correct_widths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    test_df['suitable_widths'] = test_df['correct_widths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Fallback: if correct_widths is empty or NaN, use best_width
    train_df['suitable_widths'] = train_df.apply(
        lambda row: row['suitable_widths'] if isinstance(row['suitable_widths'], list) and len(row['suitable_widths']) > 0 else [int(row['best_width'])],
        axis=1
    )
    test_df['suitable_widths'] = test_df.apply(
        lambda row: row['suitable_widths'] if isinstance(row['suitable_widths'], list) and len(row['suitable_widths']) > 0 else [int(row['best_width'])],
        axis=1
    )

    # Construct question_word_flag: 1 if any question word present, else 0
    def make_qword_flag(df):
        w = df['question_word_what'].fillna(0).astype(int)
        h = df['question_word_how'].fillna(0).astype(int)
        y = df['question_word_why'].fillna(0).astype(int)
        return (w | h | y)

    train_df['question_word_flag'] = make_qword_flag(train_df)
    test_df['question_word_flag'] = make_qword_flag(test_df)

    y_train = train_df['best_width'].values
    y_test = test_df['best_width'].values
    suitable_test = test_df['suitable_widths'].tolist()

    print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")

    # Stats on suitable_widths
    avg_sw = np.mean([len(sw) for sw in suitable_test])
    print(f"  Avg suitable widths per test sample: {avg_sw:.2f}")
    print(f"  Width distribution (test): {dict(pd.Series(y_test).value_counts().sort_index())}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    label_map = {w: i for i, w in enumerate(sorted(WIDTHS_ALL))}
    reverse_map = {i: w for w, i in label_map.items()}
    y_tr_xgb = np.array([label_map[w] for w in y_train])

    # ════════════════════════════════════════════════════════════
    #  Prepare features
    # ════════════════════════════════════════════════════════════
    X_train_raw = train_df[TEXT_FEATURES].fillna(0).values
    X_test_raw = test_df[TEXT_FEATURES].fillna(0).values

    print(f"  Feature dim: {X_train_raw.shape[1]}")

    # ════════════════════════════════════════════════════════════
    #  RF / LDA (raw features)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  10 features (text only) — RF / LDA")
    print("=" * 70)

    # RF (raw)
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_raw, y_train)
    y_pred_rf = rf.predict(X_test_raw)
    all_results["RF_raw"] = evaluate("RF(raw)", y_test, y_pred_rf, suitable_test)

    # XGB (raw)
    print()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1
    )
    xgb.fit(X_train_raw, y_tr_xgb)
    y_pred_xgb = np.array([reverse_map[i] for i in xgb.predict(X_test_raw)])
    all_results["XGB_raw"] = evaluate("XGB(raw)", y_test, y_pred_xgb, suitable_test)

    # LDA (raw)
    print()
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_raw, y_train)
    y_pred_lda = lda.predict(X_test_raw)
    all_results["LDA_raw"] = evaluate("LDA(raw)", y_test, y_pred_lda, suitable_test)

    # Save best model (LDA)
    model_path = OUTPUT_DIR / "lda_width_predictor.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': lda, 'features': TEXT_FEATURES, 'results': all_results['LDA_raw']}, f)
    print(f"\n  Best model saved: {model_path}")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Model':<12s} {'strict':>7} {'relaxed':>8} {'pm1':>7}")
    print("  " + "-" * 38)
    for name, r in all_results.items():
        print(f"  {name:<12s} {r['strict_acc']:>7.4f} {r['relaxed_acc']:>8.4f} {r['pm1_acc']:>7.4f}")

    # Save
    results_path = OUTPUT_DIR / "relaxed_acc_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
