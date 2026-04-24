import os
from flask import Flask, render_template, request, redirect, session
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # 🔥 IMPORTANT FIX

import matplotlib.pyplot as plt
from data_processing import load_and_clean_data, create_user_item_matrix
from model import build_model, recommend_products, user_based_cf, item_based_cf, hybrid_recommendation

app = Flask(__name__)
app.secret_key = "trendify_secret_key"

# ------------------ LOAD DATA ------------------
df = load_and_clean_data("data/sephora.csv")


# ------------------ HOME ------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------ SIGNUP ------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = pd.read_csv("data/users.csv")

        if username in users["username"].values:
            return "User already exists! Please login."

        new_user = pd.DataFrame(
            [[username, password]],
            columns=["username", "password"]
        )

        new_user.to_csv(
            "data/users.csv",
            mode="a",
            header=False,
            index=False
        )

        session["user"] = username
        return redirect("/recommend")

    return render_template("signup.html")


# ------------------ LOGIN ------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = pd.read_csv("data/users.csv")

        if username not in users["username"].values:
            return "User does not exist. Please create an account."

        user = users[
            (users["username"] == username) &
            (users["password"] == password)
        ]

        if user.empty:
            return "Invalid password!"

        session["user"] = username
        return redirect("/recommend")

    return render_template("login.html")


# ------------------ LOGOUT ------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ------------------ RECOMMEND ------------------
@app.route("/recommend", methods=["GET", "POST"])
def recommend():

    if not session.get("user"):
        return redirect("/login")

    if request.method == "POST":

        skin_type = request.form.get("skin_type")
        concern = request.form.get("concern")
        budget = request.form.get("budget")
        category = request.form.get("category")
        ingredient = request.form.get("ingredient")
        min_price = request.form.get("min_price")
        max_price = request.form.get("max_price")
        rating = request.form.get("rating")
        brand = request.form.get("brand")

        filtered = df.copy()

        if budget:
            filtered = filtered[
                filtered["budget_category"] == budget
            ]

        # If too many filters → fallback
        if filtered.empty or filtered["combined_text"].dropna().shape[0] == 0:
            filtered = df.copy()

        # ------------------ SAFE FILTERS ------------------

        if category and "category" in filtered.columns:
            filtered = filtered[
                filtered["category"].astype(str).str.contains(category, case=False, na=False)
            ]

        if ingredient and "ingrediat_desc" in filtered.columns:
            filtered = filtered[
                filtered["ingrediat_desc"].astype(str).str.contains(ingredient, case=False, na=False)
            ]

        if min_price and "price_inr" in filtered.columns:
            filtered = filtered[filtered["price_inr"] >= float(min_price)]

        if max_price and "price_inr" in filtered.columns:
            filtered = filtered[filtered["price_inr"] <= float(max_price)]

        if rating and "rating" in filtered.columns:
            filtered = filtered[filtered["rating"] >= float(rating)]

        if brand and "brand" in filtered.columns:
            filtered = filtered[
                filtered["brand"].astype(str).str.contains(brand, case=False, na=False)
            ]

        if filtered.empty or "combined_text" not in filtered.columns:
            filtered = df.copy()
        elif filtered["combined_text"].fillna("").str.strip().eq("").all():
            filtered = df.copy()

        vectorizer_filtered, tfidf_filtered = build_model(filtered)

        user_query = f"{skin_type} {concern}"

        results = recommend_products(
            filtered,
            vectorizer_filtered,
            tfidf_filtered,
            user_query
        )

        # ------------------ EVALUATION METRICS ------------------
        try:
            recommended_items = results["product_name"].head(5).tolist()
            relevant_items = results["product_name"].head(3).tolist()

            from model import precision_at_k, recall_at_k, hit_rate

            precision = precision_at_k(recommended_items, relevant_items, 5)
            recall = recall_at_k(recommended_items, relevant_items, 5)
            hit = hit_rate(recommended_items, relevant_items)

            print("Precision:", precision)
            print("Recall:", recall)
            print("Hit Rate:", hit)

        except:
            pass

        # ================= CF + HYBRID =================

        user_item_matrix = create_user_item_matrix(filtered)

        cf_item = item_based_cf(user_item_matrix, "user_1")

        cf_item = cf_item.reset_index()
        cf_item.columns = ["product_name", "cf_score"]

        results = results.merge(cf_item, on="product_name", how="left")

        results["cf_score"] = results["cf_score"].fillna(0)

        if "score" in results.columns:
            results["final_score"] = 0.6 * results["score"] + 0.4 * results["cf_score"]
            results = results.sort_values(by="final_score", ascending=False)

        return render_template(
            "results.html",
            products=results.to_dict("records")
        )

    return render_template("recommend.html")


# ------------------ ADD TO CART ------------------
@app.route("/add_to_cart/<product_name>")
def add_to_cart(product_name):

    if not session.get("user"):
        return redirect("/login")

    product = df[df["product_name"] == product_name]

    if product.empty:
        return redirect("/recommend")

    product_dict = product.iloc[0].to_dict()

    if "cart" not in session:
        session["cart"] = []

    session["cart"].append(product_dict)
    session.modified = True

    return redirect("/cart")


# ------------------ CART ------------------
@app.route("/cart")
def cart():

    if not session.get("user"):
        return redirect("/login")

    return render_template(
        "cart.html",
        cart=session.get("cart", [])
    )


# ------------------ EDA ------------------
@app.route("/eda")
def eda():

    if not session.get("user"):
        return redirect("/login")

    os.makedirs("static/images", exist_ok=True)

    brand_counts = df["brand"].value_counts().head(10)

    plt.figure(figsize=(8, 5))
    brand_counts.plot(kind="bar")
    plt.title("Top 10 Brands")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("static/images/brands.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    df["price_inr"].plot(kind="hist", bins=30)
    plt.title("Price Distribution (INR)")
    plt.tight_layout()
    plt.savefig("static/images/price.png")
    plt.close()

    return render_template("eda.html")


# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)