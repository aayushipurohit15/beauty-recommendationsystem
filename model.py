from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_model(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, tfidf_matrix

def recommend_products(df, vectorizer, tfidf_matrix, user_query, top_n=8):
    user_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df["similarity_score"] = similarity

    df["final_score"] = (
        0.5 * df["similarity_score"] +
        0.3 * df["popularity_score"]
    )

    return df.sort_values(by="final_score", ascending=False).head(top_n)

# ===========================
# COLLABORATIVE FILTERING MODELS
# ===========================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# USER-BASED CF
# ---------------------------
def user_based_cf(user_item_matrix, target_user, top_k=5):

    user_similarity = cosine_similarity(user_item_matrix)

    similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    similar_users = similarity_df[target_user].sort_values(ascending=False)[1:6]

    weighted_scores = np.dot(similar_users.values, user_item_matrix.loc[similar_users.index])

    recommendations = pd.Series(weighted_scores, index=user_item_matrix.columns)
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(top_k)


# ---------------------------
# ITEM-BASED CF
# ---------------------------
def item_based_cf(user_item_matrix, target_user, top_k=5):

    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    # Transpose matrix
    item_matrix = user_item_matrix.T

    # Compute similarity
    item_similarity = cosine_similarity(item_matrix)

    similarity_df = pd.DataFrame(
        item_similarity,
        index=item_matrix.index,
        columns=item_matrix.index
    )

    # Convert user row to numpy (safe)
    user_data = user_item_matrix.loc[target_user].to_numpy()

    scores = {}

    # Loop safely using index
    for idx in range(len(user_item_matrix.columns)):

        item = user_item_matrix.columns[idx]

        # Only check scalar value
        if user_data[idx] == 0:
            similar_items = similarity_df.iloc[:, idx].to_numpy()

            score = np.dot(similar_items, user_data)

            scores[item] = score

    # Convert to series
    recommendations = pd.Series(scores)

    # Sort safely
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(top_k)

# ---------------------------
# EVALUATION METRICS
# ---------------------------
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / len(relevant)


def hit_rate(recommended, relevant):
    return int(len(set(recommended) & set(relevant)) > 0)


# ---------------------------
# HYBRID MODEL
# ---------------------------
def hybrid_recommendation(content_scores, cf_scores, alpha=0.6):

    final_scores = {}

    for item in content_scores.index:
        content_score = content_scores.get(item, 0)
        cf_score = cf_scores.get(item, 0)

        final_scores[item] = alpha * content_score + (1 - alpha) * cf_score

    return pd.Series(final_scores).sort_values(ascending=False)
