# SmartCart

SmartCart is a runnable FastAPI e-commerce demo with SQLite, JWT authentication, synthetic customer/order/activity data, purchase prediction, customer segmentation, recommendations, admin analytics, and a browser storefront.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m smartcart.data
python -m smartcart.train
uvicorn smartcart.main:app --reload
```

Open `http://127.0.0.1:8000/` for the storefront or `http://127.0.0.1:8000/docs` for Swagger UI. Startup creates `data/smartcart.db`, CSV datasets in `data/`, and `artifacts/purchase_model.pkl` plus `artifacts/customer_segments.pkl` if missing. Seeded demo accounts are `maya@smartcart.com` / `Customer123!` and `admin@smartcart.com` / `Admin123!`. Set `SMARTCART_DB` to use a different SQLite file and `SMARTCART_SECRET_KEY` to replace the development JWT secret.

## Dependencies and ML notes

FastAPI and Uvicorn serve the API; SQLite (Python standard library) provides persistence; Pandas and NumPy generate and prepare data; scikit-learn supplies preprocessing, Logistic Regression, Decision Tree, Random Forest, K-Means, and silhouette scoring; Joblib stores fitted pipelines; Pydantic validates payloads; `python-jose` and `pwdlib` implement JWTs and password hashing; HTTPX and pytest support API tests. Matplotlib creates the EDA charts.

Before training, remove duplicate rows, validate IDs against customer/product tables, coerce dates and numeric columns, reject invalid ages/prices/quantities, impute missing ratings, and inspect missingness and label balance. The generated training matrix is numeric and complete; its fitted `StandardScaler` is part of each model pipeline. The evaluation script uses a stratified holdout and reports accuracy, precision, recall, F1, confusion matrix, and ROC-AUC. The model is selected by holdout F1 (ROC-AUC breaks ties), not training accuracy. These synthetic metrics are for demonstrating the workflow, not production claims.

Customer segmentation scales purchases, spending, average order value, visits, and purchase frequency. Training evaluates silhouette scores for K=2..6 and selects the strongest valid score; cluster names are assigned by relative customer value. For real data, validate the chosen K and segment stability with domain review and temporal holdouts.

## Data and analysis

`python -m smartcart.data` writes `customers.csv` (100 rows), `products.csv` (50), `orders.csv` (500), and `customer_activity.csv` (all customer/product pairs) to `data/`. The generator uses a fixed seed. `python -m scripts.analyze_data` prints sales/customer summaries and writes category revenue, monthly revenue, and product-rating charts under `data/eda/`.

## APIs

Customer and product CRUD, cart CRUD, orders and customer order history, registration/login, `/predict-purchase`, `/recommendations/{customer_id}`, `/customer-segment/{customer_id}`, `/customer-dashboard/{customer_id}`, and admin-only `/admin/sales-summary`, `/admin/top-products`, `/admin/top-customers`, `/admin/category-sales`, `/admin/customer-segments`, and `/admin/low-stock-products` are available. Admin endpoints require the seeded administrator bearer token; public registration always creates customer-role accounts.

## Demo flow

Register through the storefront or `POST /register`, log in, browse products, request recommendations, add an item to the cart, then check out. Checkout creates an order and order items, reduces stock, and refreshes the customer's purchase totals. The dashboard then exposes current recommendations, purchase probabilities, and the latest customer segment. Sign in as the demo admin to view sales analytics.

## Verify

```bash
pytest -q
```