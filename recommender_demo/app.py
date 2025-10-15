from fastapi import FastAPI, HTTPException
import numpy as np, pandas as pd, os
from typing import List

BASE = os.path.dirname(__file__)
DATA_DIR = BASE  # data files are in same directory
MODEL_PATH = os.path.join(BASE, "svd_model.npz")

app = FastAPI(title="Recommender Demo")

# load dataset and model at startup
_products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
_model = np.load(MODEL_PATH, allow_pickle=True)
_user_factors = _model["user_factors"]
_item_factors = _model["item_factors"]
_users = _model["users"]
_products_ids = _model["products"]

def recommend_for_user(user_id: int, top_k: int = 10):
    try:
        uidx = int(list(_users).index(user_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="User not found")
    uvec = _user_factors[uidx]
    scores = _item_factors.dot(uvec)
    top_idx = list(scores.argsort()[::-1][:top_k])
    top_products = [_products.loc[_products["product_id"]==int(_products_ids[i])].to_dict(orient="records")[0] for i in top_idx]
    return [{"product": p, "score": float(scores[i])} for i,p in zip(top_idx, top_products)]

@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = 10):
    return recommend_for_user(user_id, top_k=k)

@app.post("/explain/")
def explain(user_id: int, product_id: int, user_history: List[int] = []):
    prod = _products.loc[_products["product_id"]==product_id]
    if prod.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    prod = prod.iloc[0]
    reasons = []
    try:
        hist_cats = [ _products.loc[_products["product_id"]==pid]["category"].values[0] for pid in user_history if pid in list(_products["product_id"]) ]
    except Exception:
        hist_cats = []
    if prod["category"] in hist_cats:
        reasons.append(f"You have interacted with products in the category '{prod['category']}', so similar items are suggested.")
    reasons.append(f"This product's description: {prod['description'][:120]}...")
    reasons.append("Model similarity between you and this product is high based on latent features from past interactions.")
    explanation = " ".join(reasons)
    return {"user_id": user_id, "product_id": int(product_id), "explanation": explanation}
