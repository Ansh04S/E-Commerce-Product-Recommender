# E-commerce Recommender Demo

This is a self-contained demo project created automatically for your assignment.

Files:
- products.csv : synthetic product catalog
- interactions.csv : synthetic user interactions (implicit weights)
- svd_model.npz : trained SVD model (user & item factors)
- app.py : FastAPI app exposing /recommend/{user_id} and /explain endpoints
- requirements.txt : pip install -r requirements.txt

Run locally (example):
1. Create virtualenv and install:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Start server:
   uvicorn app:app --reload --port 8000

3. Test:
   GET http://localhost:8000/recommend/1?k=5
   POST http://localhost:8000/explain/  with JSON body:
     { "user_id": 1, "product_id": 5, "user_history": [3,7,10] }

Notes:
- The explanation endpoint returns a simple template. To integrate a real LLM, add your OpenAI (or other) API call inside explain(), and compose a prompt like:
  "Explain why product {product_title} is recommended to user {user_id} based on their history: {history} and the product metadata: {metadata}."

- This demo uses TruncatedSVD for a quick collaborative approach; for production, consider LightFM, implicit ALS, or neural recommenders, and add caching, batching, authentication, and rate limits.

Good luck with your recruitment assignment!
