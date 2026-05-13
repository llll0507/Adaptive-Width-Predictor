# -*- coding: utf-8 -*-
"""
Extract 12-dimensional features from question text.
"""
import re
from pathlib import Path
import pandas as pd
import spacy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

FEATURE_COLUMNS = [
    "length_ratio",
    "num_entities",
    "sentence_complexity",
    "keyword_density",
    "sentiment_strength",
    "dependency_depth",
    "multi_entity_flag",
    "pronoun_density",
    "verb_density",
    "question_word_flag",
]

# Question words for features
WHAT_WORDS = {"what", "which", "who", "whom", "whose"}
HOW_WORDS = {"how", "how many", "how much", "how long", "how often"}
WHY_WORDS = {"why", "what for", "for what reason"}

def extract_features(text):
    """Extract 12 features from question text."""
    doc = nlp(text.lower())
    
    # 1. length_ratio: question length / avg question length (will normalize later)
    length_ratio = len(text.split())
    
    # 2. num_entities: number of named entities
    num_entities = len(doc.ents)
    
    # 3. sentence_complexity: number of clauses (conj + relcl)
    sentence_complexity = sum(
        1 for token in doc if token.dep_ in {"conj", "relcl", "ccomp", "xcomp"}
    )
    
    # 4. keyword_density: ratio of nouns+adjs to total tokens
    content_words = sum(
        1 for token in doc if token.pos_ in {"NOUN", "ADJ", "PROPN"}
    )
    keyword_density = content_words / max(len(doc), 1)
    
    # 5. sentiment_strength: VADER compound score (absolute value)
    sentiment_strength = abs(analyzer.polarity_scores(text)['compound'])
    
    # 6. dependency_depth: max dependency tree depth
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
    
    # 7. multi_entity_flag: 1 if more than 2 entities, else 0
    multi_entity_flag = 1 if num_entities > 2 else 0
    
    # 8. pronoun_density: ratio of pronouns
    pronoun_density = sum(
        1 for token in doc if token.pos_ == "PRON"
    ) / max(len(doc), 1)
    
    # 9. verb_density: ratio of verbs
    verb_density = sum(
        1 for token in doc if token.pos_ == "VERB"
    ) / max(len(doc), 1)
    
    # 10. Question word flag (any question word present)
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

def process_file(input_path, output_path):
    """Process a CSV file and add 10 features."""
    print(f"Processing: {input_path}")
    df = pd.read_csv(input_path)
    
    # Extract features
    features = df["question"].apply(extract_features).apply(pd.Series)
    df = pd.concat([df, features], axis=1)
    
    # Normalize length_ratio by dataset mean
    mean_length = df["length_ratio"].mean()
    df["length_ratio"] = df["length_ratio"] / max(mean_length, 1)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    feature_dir = base_dir / "feature_data"
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train, val, test
    process_file(base_dir / "train_final.csv", feature_dir / "train_with_features.csv")
    process_file(base_dir / "val_final.csv", feature_dir / "val_with_features.csv")
    process_file(base_dir / "test_final.csv", feature_dir / "test_with_features.csv")
    
    print("\n✓ All features extracted successfully!")
