import pandas as pd
import numpy as np

def clean_image_url(image_string):
    if pd.isna(image_string):
        return None

    first_image = image_string.split("~")[0]

    first_image = first_image.replace(
        "https://www.sephora.comhttps://www.sephora.com",
        "https://www.sephora.com"
    )

    return first_image.strip()


def load_and_clean_data(path):
    df = pd.read_csv(path)

    df = df.drop_duplicates()

    df["ingrediat_desc"] = df["ingrediat_desc"].fillna("").str.lower()
    df["about_product"] = df["about_product"].fillna("").str.lower()

    df["price"] = df["price"].astype(str)
    df["price"] = df["price"].str.replace("$", "", regex=False)
    df["price"] = df["price"].str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(0)

    # Convert USD to INR
    df["price_inr"] = df["price"] * 83

    df["reviews_count"] = df["reviews_count"].fillna(1)
    df["likes_count"] = df["likes_count"].fillna(1)

    df["image_url"] = df["images"].apply(clean_image_url)

    df["combined_text"] = df["ingrediat_desc"] + " " + df["about_product"]

    # ===========================
    # ✅ UPDATED CATEGORY LOGIC
    # ===========================
    def detect_category(text):
        text = str(text).lower()

        if "lip" in text:
            return "lip"
        elif "eye" in text:
            return "eye"
        elif "face" in text:
            return "face"
        elif "skin" in text or "skincare" in text:
            return "skincare"
        else:
            return "other"

    df["category"] = df["combined_text"].apply(detect_category)
    # ===========================

    df["popularity_score"] = np.log1p(df["reviews_count"] + df["likes_count"])

    def budget_category(price):
        if price <= 25:
            return "Low"
        elif price <= 60:
            return "Medium"
        else:
            return "High"

    df["budget_category"] = df["price"].apply(budget_category)

    return df


# ===========================
# COLLABORATIVE FILTERING DATA
# ===========================

def create_user_item_matrix(df):
    num_users = 20
    num_items = len(df)

    np.random.seed(42)

    interaction_matrix = np.random.randint(0, 2, size=(num_users, num_items))

    user_item_df = pd.DataFrame(
        interaction_matrix,
        index=[f"user_{i}" for i in range(num_users)],
        columns=df["product_name"]
    )

    return user_item_df