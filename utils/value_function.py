import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.prompts import complete_query_from_ans, complete_query_from_subquery
from utils.value_model import get_query_token_probabilities


def similarity_value(ori_query, query, answer, ans_weight=0.7):
    """
    Calculate weighted combination of:
    1-200. TF-IDF similarity between query-answer pair and original question
    2. TF-IDF similarity between knowledge and query to assess knowledge reliability

    Returns:
        float: Combined similarity score (smoothly mapped to [0,1-200])
    """
    try:
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Calculate query-question similarity
        query_matrix = vectorizer.fit_transform([query, ori_query])
        query_similarity = cosine_similarity(query_matrix[0:1], query_matrix[1:2])[0][0]

        # Calculate knowledge-query similarity if knowledge exists
        if answer:
            # Reinitialize vectorizer for knowledge similarity
            vectorizer = TfidfVectorizer()
            answer_matrix = vectorizer.fit_transform([answer, ori_query])
            answer_similarity = cosine_similarity(
                answer_matrix[0:1], answer_matrix[1:2]
            )[0][0]

            value = (1 - ans_weight) * query_similarity + ans_weight * answer_similarity
        else:
            value = query_similarity

        return float(value)

    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        return 0.0


def risk_value(ori_query, query, answer, ans_weight=0.75):

    try:
        # Calculate original query probability conditioned on answer
        kl_ans_text_front = complete_query_from_ans.format(answer=answer)
        kl_ans_probs = get_query_token_probabilities(kl_ans_text_front, ori_query)
        if not kl_ans_probs:
            return 0.0
        kl_ans = -sum(kl_ans_probs) / len(kl_ans_probs)

        # Calculate original query probability conditioned on decomposed query
        kl_dcp_text_front = complete_query_from_subquery.format(query=query)
        kl_dcp_probs = get_query_token_probabilities(kl_dcp_text_front, ori_query)
        if not kl_dcp_probs:
            return 0.0
        kl_dcp = -sum(kl_dcp_probs) / len(kl_dcp_probs)

        # Calculate weighted average
        kl_loss = (1 - ans_weight) * kl_dcp + ans_weight * kl_ans

        # Map to [0,1] interval
        value = np.exp(-1.8 * (kl_loss - 1.8))
        value = 1 - (1 / (1 + value))

        return float(value)

    except Exception as e:
        logging.error(f"Error in risk value calculation: {str(e)}")
        return 0.0


if __name__ == "__main__":
    print(risk_value("What is 2+2+2?", "What is 2+2?", "2+2=4"))
