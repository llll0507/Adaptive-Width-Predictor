import json
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_retriever_cache = {}


@lru_cache(maxsize=4096)
def get_retriever(data_path, data_idx):
    """Build TF-IDF retrieval index from dataset (cached by question)"""
    if (data_path, data_idx) in _retriever_cache:
        return _retriever_cache[(data_path, data_idx)]

    try:
        with open(data_path, encoding="utf-8") as f:
            database = json.load(f)
            context = database[data_idx]["context"]

        # Build document list, each = title + all facts concatenated
        documents = []
        for ele in context:
            theme = ele[0]
            infos = ele[1]
            # infos may be list or str
            if isinstance(infos, list):
                facts_text = " ".join(str(s) for s in infos)
            else:
                facts_text = str(infos)
            documents.append({
                "keyword": theme,
                "facts": facts_text,
                "text": f"{theme} {facts_text}",
            })

        if not documents:
            return None

        # Build TF-IDF matrix
        corpus = [doc["text"] for doc in documents]
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(corpus)

        retriever_data = {
            "documents": documents,
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
        }

        _retriever_cache[(data_path, data_idx)] = retriever_data
        return retriever_data

    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


def retrieve(query, data_path, data_idx, topk=5):

    try:
        print("Starting retrieval process...")

        retriever_data = get_retriever(data_path, data_idx)
        if retriever_data is None:
            print("Failed to initialize retriever")
            return ""

        documents   = retriever_data["documents"]
        vectorizer  = retriever_data["vectorizer"]
        tfidf_matrix = retriever_data["tfidf_matrix"]
        N = len(documents)

        if N == 0:
            print("No documents to search")
            return ""

        # Convert query to TF-IDF vector
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix)[0]

        # Take top-k (don't filter score=0, avoid returning empty when all zeros)
        k = min(topk, N)
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            doc = documents[idx]
            results.append(f"{doc['keyword']}: {doc['facts']}")

        print(f"Search completed, found {len(results)} documents")
        return "\n".join(results)

    except Exception as e:
        print(f"Error during retrieval: {str(e)}")
        return ""
