"""FastAPI routes for the SmartCart storefront and ML services."""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import sqlite3

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from smartcart.data import DATA_DIR, ROOT
from smartcart.database import connect_db, init_db, seed_database
from smartcart.security import (JWTError, create_access_token, decode_access_token,
                                hash_password, verify_password)
from smartcart.train import ARTIFACT_DIR, FEATURES, SEGMENT_FEATURES, train_models

STATIC_DIR = ROOT / "static"
PURCHASE_MODEL = None
SEGMENT_MODEL = None
bearer = HTTPBearer(auto_error=False)


class RegistrationIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    age: int = Field(ge=18, le=100)
    city: str = Field(min_length=1, max_length=80)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class CustomerCreate(BaseModel):
    full_name: str = Field(default="SmartCart Customer", min_length=2, max_length=100)
    email: EmailStr | None = None
    age: int = Field(ge=18, le=100)
    gender: str = Field(default="Not specified", max_length=40)
    city: str = Field(default="Unknown", min_length=1, max_length=80)
    registration_date: date = Field(default_factory=date.today)
    total_purchases: int = Field(default=0, ge=0)
    total_spending: float = Field(default=0, ge=0)
    average_order_value: float = Field(default=0, ge=0)
    website_visits: int = Field(default=0, ge=0)


class CustomerUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=2, max_length=100)
    age: int | None = Field(default=None, ge=18, le=100)
    gender: str | None = Field(default=None, max_length=40)
    city: str | None = Field(default=None, min_length=1, max_length=80)
    website_visits: int | None = Field(default=None, ge=0)


class ProductIn(BaseModel):
    product_name: str = Field(min_length=2, max_length=160)
    category: str = Field(min_length=1, max_length=80)
    price: float = Field(gt=0, le=1_000_000)
    rating: float = Field(default=0, ge=0, le=5)
    stock: int = Field(default=0, ge=0)
    discount: float = Field(default=0, ge=0, le=100)


class CartIn(BaseModel):
    customer_id: int = Field(gt=0)
    product_id: int = Field(gt=0)
    quantity: int = Field(ge=1, le=100)


class CartUpdate(BaseModel):
    quantity: int = Field(ge=1, le=100)


class OrderIn(BaseModel):
    payment_method: str = Field(default="Card", min_length=2, max_length=40)


class PredictionIn(BaseModel):
    customer_id: int = Field(gt=0)
    product_id: int = Field(gt=0)


class RecommendationOut(BaseModel):
    product_id: int
    product_name: str
    score: float


class OrderItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    product_id: int
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)


def _load_artifacts() -> None:
    global PURCHASE_MODEL, SEGMENT_MODEL
    purchase_path = ARTIFACT_DIR / "purchase_model.pkl"
    segment_path = ARTIFACT_DIR / "customer_segments.pkl"
    if not purchase_path.exists() or not segment_path.exists():
        train_models()
    PURCHASE_MODEL = joblib.load(purchase_path)
    SEGMENT_MODEL = joblib.load(segment_path)


def _seed_demo_users() -> None:
    with connect_db() as connection:
        connection.execute(
            "UPDATE customers SET full_name = ?, email = ? WHERE customer_id = 1 AND email IS NULL",
            ("Maya Chen", "maya@smartcart.com"),
        )
        customer = connection.execute("SELECT customer_id FROM customers WHERE email = ?", ("maya@smartcart.com",)).fetchone()
        if customer:
            connection.execute(
                "INSERT OR IGNORE INTO users(customer_id, email, password_hash, role) VALUES (?, ?, ?, 'customer')",
                (customer["customer_id"], "maya@smartcart.com", hash_password("Customer123!")),
            )
        connection.execute(
            "INSERT OR IGNORE INTO users(customer_id, email, password_hash, role) VALUES (NULL, ?, ?, 'admin')",
            ("admin@smartcart.com", hash_password("Admin123!")),
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not (DATA_DIR / "customers.csv").exists():
        from smartcart.data import generate_datasets
        generate_datasets()
    init_db()
    seed_database()
    _seed_demo_users()
    _load_artifacts()
    yield


app = FastAPI(title="SmartCart API", version="1.0.0", description="E-commerce APIs with purchase prediction, recommendations, and customer segments.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def _customer_or_404(connection: sqlite3.Connection, customer_id: int) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return row


def _product_or_404(connection: sqlite3.Connection, product_id: int) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return row


def _current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Bearer token required", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_access_token(credentials.credentials)
        user_id = int(claims["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    with connect_db() as connection:
        user = connection.execute("SELECT user_id, customer_id, email, role FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="Account not found")
    return dict(user)


def _admin(user: dict = Depends(_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user


def _require_owner(customer_id: int, user: dict) -> None:
    if user["role"] != "admin" and user["customer_id"] != customer_id:
        raise HTTPException(status_code=403, detail="You can only access your own account")


def _recommend(connection: sqlite3.Connection, customer_id: int, limit: int = 5) -> list[dict]:
    customer = _customer_or_404(connection, customer_id)
    purchased = {row[0] for row in connection.execute(
        "SELECT DISTINCT oi.product_id FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.customer_id = ? AND o.order_status != 'cancelled'",
        (customer_id,),
    )}
    categories = connection.execute(
        """SELECT p.category, SUM(oi.quantity) AS quantity FROM orders o
           JOIN order_items oi ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id
           WHERE o.customer_id = ? AND o.order_status != 'cancelled' GROUP BY p.category""",
        (customer_id,),
    ).fetchall()
    preferences = {row["category"]: row["quantity"] for row in categories}
    total_preferences = max(sum(preferences.values()), 1)
    popularity = {row["product_id"]: row["sales"] for row in connection.execute(
        "SELECT product_id, SUM(quantity) AS sales FROM order_items GROUP BY product_id"
    )}
    max_popularity = max(popularity.values(), default=1)
    products = connection.execute("SELECT * FROM products WHERE stock > 0 ORDER BY rating DESC").fetchall()
    results = []
    for product in products:
        if product["product_id"] in purchased:
            continue
        preference = preferences.get(product["category"], 0) / total_preferences
        score = 0.58 * product["rating"] / 5 + 0.32 * preference + 0.10 * popularity.get(product["product_id"], 0) / max_popularity
        results.append({"product_id": product["product_id"], "product_name": product["product_name"], "score": round(float(score), 4)})
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:limit]


def _features(connection: sqlite3.Connection, customer_id: int, product_id: int) -> dict:
    customer = _customer_or_404(connection, customer_id)
    product = _product_or_404(connection, product_id)
    activity = connection.execute("SELECT * FROM customer_activity WHERE customer_id = ? AND product_id = ?", (customer_id, product_id)).fetchone()
    previous = connection.execute(
        "SELECT COALESCE(SUM(oi.quantity), 0) FROM orders o JOIN order_items oi ON oi.order_id = o.order_id WHERE o.customer_id = ? AND oi.product_id = ? AND o.order_status != 'cancelled'",
        (customer_id, product_id),
    ).fetchone()[0]
    if activity is None:
        activity_values = {"product_views": 0, "cart_additions": 0, "wishlist_additions": 0, "time_spent": 0}
    else:
        activity_values = dict(activity)
    activity_values["previous_purchases"] = max(int(activity_values.get("previous_purchases", 0)), int(previous))
    return {
        "customer_age": customer["age"], "total_purchases": customer["total_purchases"],
        "total_spending": customer["total_spending"], "average_order_value": customer["average_order_value"],
        "website_visits": customer["website_visits"], "product_price": product["price"],
        "product_rating": product["rating"], "product_views": activity_values["product_views"],
        "cart_additions": activity_values["cart_additions"], "wishlist_additions": activity_values["wishlist_additions"],
        "previous_purchases": activity_values["previous_purchases"], "time_spent": activity_values["time_spent"],
    }


def _segment(connection: sqlite3.Connection, customer: sqlite3.Row) -> str:
    if SEGMENT_MODEL is None:
        raise HTTPException(status_code=503, detail="Customer segmentation model is unavailable")
    age_days = max((date.today() - date.fromisoformat(customer["registration_date"][:10])).days, 1)
    values = pd.DataFrame([{
        "total_purchases": customer["total_purchases"], "total_spending": customer["total_spending"],
        "average_order_value": customer["average_order_value"], "website_visits": customer["website_visits"],
        "purchase_frequency": customer["total_purchases"] / age_days * 365,
    }], columns=SEGMENT_FEATURES)
    cluster = int(SEGMENT_MODEL["model"].predict(SEGMENT_MODEL["scaler"].transform(values))[0])
    return SEGMENT_MODEL["segment_names"][cluster]


@app.get("/", include_in_schema=False)
def storefront():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegistrationIn):
    with connect_db() as connection:
        try:
            cursor = connection.execute(
                """INSERT INTO customers(full_name, email, age, city, registration_date)
                   VALUES (?, ?, ?, ?, ?)""",
                (payload.full_name, str(payload.email).lower(), payload.age, payload.city, date.today().isoformat()),
            )
            customer_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO users(customer_id, email, password_hash, role) VALUES (?, ?, ?, 'customer')",
                (customer_id, str(payload.email).lower(), hash_password(payload.password)),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="An account with this email already exists")
        user = connection.execute("SELECT user_id, customer_id, email, role FROM users WHERE user_id = last_insert_rowid()").fetchone()
    token = create_access_token(user["user_id"], user["role"])
    return {"access_token": token, "token_type": "bearer", "customer": {"customer_id": customer_id, "full_name": payload.full_name, "email": str(payload.email)}}


@app.post("/login")
def login(payload: LoginIn):
    with connect_db() as connection:
        user = connection.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (str(payload.email),)).fetchone()
    if user is None or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password", headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": create_access_token(user["user_id"], user["role"]), "token_type": "bearer", "role": user["role"], "customer_id": user["customer_id"]}


@app.post("/customers", status_code=status.HTTP_201_CREATED)
def create_customer(payload: CustomerCreate):
    values = payload.model_dump()
    values["registration_date"] = values["registration_date"].isoformat()
    with connect_db() as connection:
        try:
            cursor = connection.execute(
                """INSERT INTO customers(full_name, email, age, gender, city, registration_date,
                   total_purchases, total_spending, average_order_value, website_visits)
                   VALUES (:full_name, :email, :age, :gender, :city, :registration_date,
                   :total_purchases, :total_spending, :average_order_value, :website_visits)""", values,
            )
            customer_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="A customer with this email already exists")
        return dict(_customer_or_404(connection, customer_id))


@app.get("/customers")
def list_customers(limit: int = Query(default=100, ge=1, le=500), offset: int = Query(default=0, ge=0)):
    with connect_db() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM customers ORDER BY customer_id LIMIT ? OFFSET ?", (limit, offset))]


@app.get("/customers/{customer_id}")
def get_customer(customer_id: int):
    with connect_db() as connection:
        return dict(_customer_or_404(connection, customer_id))


@app.put("/customers/{customer_id}")
def update_customer(customer_id: int, payload: CustomerUpdate):
    values = payload.model_dump(exclude_unset=True)
    if not values:
        raise HTTPException(status_code=400, detail="No customer fields supplied")
    with connect_db() as connection:
        _customer_or_404(connection, customer_id)
        assignments = ", ".join(f"{column} = ?" for column in values)
        connection.execute(f"UPDATE customers SET {assignments} WHERE customer_id = ?", [*values.values(), customer_id])
        return dict(_customer_or_404(connection, customer_id))


@app.delete("/customers/{customer_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_customer(customer_id: int):
    with connect_db() as connection:
        _customer_or_404(connection, customer_id)
        connection.execute("DELETE FROM customers WHERE customer_id = ?", (customer_id,))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/products", status_code=status.HTTP_201_CREATED)
def create_product(payload: ProductIn):
    with connect_db() as connection:
        cursor = connection.execute(
            "INSERT INTO products(product_name, category, price, rating, stock, discount) VALUES (?, ?, ?, ?, ?, ?)",
            tuple(payload.model_dump().values()),
        )
        return dict(_product_or_404(connection, cursor.lastrowid))


@app.get("/products")
def list_products(search: str | None = None, category: str | None = None, limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0)):
    clauses, parameters = [], []
    if search:
        clauses.append("(product_name LIKE ? OR category LIKE ?)")
        parameters.extend([f"%{search}%", f"%{search}%"])
    if category:
        clauses.append("category = ? COLLATE NOCASE")
        parameters.append(category)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect_db() as connection:
        return [dict(row) for row in connection.execute(f"SELECT * FROM products {where} ORDER BY product_id LIMIT ? OFFSET ?", (*parameters, limit, offset))]


@app.get("/products/{product_id}")
def get_product(product_id: int):
    with connect_db() as connection:
        return dict(_product_or_404(connection, product_id))


@app.put("/products/{product_id}")
def update_product(product_id: int, payload: ProductIn):
    with connect_db() as connection:
        _product_or_404(connection, product_id)
        values = payload.model_dump()
        connection.execute(
            "UPDATE products SET product_name=?, category=?, price=?, rating=?, stock=?, discount=? WHERE product_id=?",
            (*values.values(), product_id),
        )
        return dict(_product_or_404(connection, product_id))


@app.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(product_id: int):
    with connect_db() as connection:
        _product_or_404(connection, product_id)
        try:
            connection.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Product has order history and cannot be deleted")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/cart", status_code=status.HTTP_201_CREATED)
def add_to_cart(payload: CartIn, user: dict = Depends(_current_user)):
    _require_owner(payload.customer_id, user)
    with connect_db() as connection:
        _customer_or_404(connection, payload.customer_id)
        product = _product_or_404(connection, payload.product_id)
        existing = connection.execute("SELECT * FROM cart WHERE customer_id=? AND product_id=?", (payload.customer_id, payload.product_id)).fetchone()
        quantity = payload.quantity + (existing["quantity"] if existing else 0)
        if quantity > product["stock"]:
            raise HTTPException(status_code=409, detail="Insufficient stock")
        if existing:
            connection.execute("UPDATE cart SET quantity = ? WHERE cart_id = ?", (quantity, existing["cart_id"]))
            cart_id = existing["cart_id"]
        else:
            cart_id = connection.execute("INSERT INTO cart(customer_id, product_id, quantity) VALUES (?, ?, ?)", (payload.customer_id, payload.product_id, quantity)).lastrowid
        connection.execute(
            """INSERT INTO customer_activity(customer_id, product_id, product_views, cart_additions)
               VALUES (?, ?, 1, 1) ON CONFLICT(customer_id, product_id) DO UPDATE SET
               product_views=product_views+1, cart_additions=cart_additions+1""",
            (payload.customer_id, payload.product_id),
        )
        return dict(connection.execute("SELECT c.*, p.product_name, p.price, p.stock FROM cart c JOIN products p USING(product_id) WHERE cart_id=?", (cart_id,)).fetchone())


@app.get("/cart/{customer_id}")
def get_cart(customer_id: int, user: dict = Depends(_current_user)):
    _require_owner(customer_id, user)
    with connect_db() as connection:
        _customer_or_404(connection, customer_id)
        return [dict(row) for row in connection.execute("SELECT c.*, p.product_name, p.price, p.discount, p.stock FROM cart c JOIN products p USING(product_id) WHERE c.customer_id=? ORDER BY c.cart_id", (customer_id,))]


@app.put("/cart/{cart_id}")
def update_cart(cart_id: int, payload: CartUpdate, user: dict = Depends(_current_user)):
    with connect_db() as connection:
        cart_item = connection.execute("SELECT * FROM cart WHERE cart_id=?", (cart_id,)).fetchone()
        if cart_item is None:
            raise HTTPException(status_code=404, detail="Cart item not found")
        _require_owner(cart_item["customer_id"], user)
        product = _product_or_404(connection, cart_item["product_id"])
        if payload.quantity > product["stock"]:
            raise HTTPException(status_code=409, detail="Insufficient stock")
        connection.execute("UPDATE cart SET quantity=? WHERE cart_id=?", (payload.quantity, cart_id))
        return dict(connection.execute("SELECT c.*, p.product_name, p.price, p.stock FROM cart c JOIN products p USING(product_id) WHERE cart_id=?", (cart_id,)).fetchone())


@app.delete("/cart/{cart_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_cart_item(cart_id: int, user: dict = Depends(_current_user)):
    with connect_db() as connection:
        cart_item = connection.execute("SELECT * FROM cart WHERE cart_id=?", (cart_id,)).fetchone()
        if cart_item is None:
            raise HTTPException(status_code=404, detail="Cart item not found")
        _require_owner(cart_item["customer_id"], user)
        connection.execute("DELETE FROM cart WHERE cart_id=?", (cart_id,))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/orders", status_code=status.HTTP_201_CREATED)
def create_order(payload: OrderIn, user: dict = Depends(_current_user)):
    if user["customer_id"] is None:
        raise HTTPException(status_code=400, detail="An admin account cannot place customer orders")
    customer_id = user["customer_id"]
    with connect_db() as connection:
        items = connection.execute("SELECT c.*, p.price, p.discount, p.stock FROM cart c JOIN products p USING(product_id) WHERE c.customer_id=?", (customer_id,)).fetchall()
        if not items:
            raise HTTPException(status_code=400, detail="Cart is empty")
        for item in items:
            if item["quantity"] > item["stock"]:
                raise HTTPException(status_code=409, detail=f"Insufficient stock for product {item['product_id']}")
        total = sum(item["quantity"] * item["price"] * (1 - item["discount"] / 100) for item in items)
        order_id = connection.execute(
            "INSERT INTO orders(customer_id, payment_method, order_status, total_amount) VALUES (?, ?, 'processing', ?)",
            (customer_id, payload.payment_method, round(total, 2)),
        ).lastrowid
        for item in items:
            unit_price = round(item["price"] * (1 - item["discount"] / 100), 2)
            connection.execute("INSERT INTO order_items(order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)", (order_id, item["product_id"], item["quantity"], unit_price))
            changed = connection.execute("UPDATE products SET stock=stock-? WHERE product_id=? AND stock>=?", (item["quantity"], item["product_id"], item["quantity"]))
            if changed.rowcount != 1:
                raise HTTPException(status_code=409, detail=f"Insufficient stock for product {item['product_id']}")
            connection.execute(
                """INSERT INTO customer_activity(customer_id, product_id, previous_purchases, purchased)
                   VALUES (?, ?, ?, 1) ON CONFLICT(customer_id, product_id) DO UPDATE SET
                   previous_purchases=previous_purchases+excluded.previous_purchases, purchased=1""",
                (customer_id, item["product_id"], item["quantity"]),
            )
        connection.execute("DELETE FROM cart WHERE customer_id=?", (customer_id,))
        connection.execute(
            """UPDATE customers SET total_purchases=(SELECT COUNT(*) FROM orders WHERE customer_id=? AND order_status!='cancelled'),
               total_spending=(SELECT COALESCE(SUM(total_amount),0) FROM orders WHERE customer_id=? AND order_status!='cancelled'),
               average_order_value=CASE WHEN (SELECT COUNT(*) FROM orders WHERE customer_id=? AND order_status!='cancelled')=0 THEN 0
               ELSE (SELECT SUM(total_amount) FROM orders WHERE customer_id=? AND order_status!='cancelled') /
               (SELECT COUNT(*) FROM orders WHERE customer_id=? AND order_status!='cancelled') END
               WHERE customer_id=?""",
            (customer_id, customer_id, customer_id, customer_id, customer_id, customer_id),
        )
        order = dict(connection.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone())
        order["items"] = [dict(row) for row in connection.execute("SELECT product_id, quantity, unit_price FROM order_items WHERE order_id=?", (order_id,))]
        return order


@app.get("/orders")
def list_orders(user: dict = Depends(_current_user)):
    with connect_db() as connection:
        if user["role"] == "admin":
            rows = connection.execute("SELECT * FROM orders ORDER BY order_date DESC")
        else:
            rows = connection.execute("SELECT * FROM orders WHERE customer_id=? ORDER BY order_date DESC", (user["customer_id"],))
        return [dict(row) for row in rows]


@app.get("/orders/{order_id}")
def get_order(order_id: int, user: dict = Depends(_current_user)):
    with connect_db() as connection:
        order = connection.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        if order is None:
            raise HTTPException(status_code=404, detail="Order not found")
        _require_owner(order["customer_id"], user)
        result = dict(order)
        result["items"] = [dict(row) for row in connection.execute("SELECT oi.product_id, p.product_name, oi.quantity, oi.unit_price FROM order_items oi JOIN products p USING(product_id) WHERE order_id=?", (order_id,))]
        return result


@app.get("/customers/{customer_id}/orders")
def customer_orders(customer_id: int, user: dict = Depends(_current_user)):
    _require_owner(customer_id, user)
    with connect_db() as connection:
        _customer_or_404(connection, customer_id)
        return [dict(row) for row in connection.execute("SELECT * FROM orders WHERE customer_id=? ORDER BY order_date DESC", (customer_id,))]


@app.post("/predict-purchase")
def predict_purchase(payload: PredictionIn):
    if PURCHASE_MODEL is None:
        raise HTTPException(status_code=503, detail="Purchase prediction model is unavailable")
    with connect_db() as connection:
        features = _features(connection, payload.customer_id, payload.product_id)
    matrix = pd.DataFrame([{key: features[key] for key in PURCHASE_MODEL["features"]}])
    probability = float(PURCHASE_MODEL["model"].predict_proba(matrix)[0, 1])
    return {"customer_id": payload.customer_id, "product_id": payload.product_id, "purchase_probability": round(probability, 4), "prediction": "Likely to Purchase" if probability >= 0.5 else "Unlikely to Purchase"}


@app.get("/recommendations/{customer_id}", response_model=dict)
def recommendations(customer_id: int):
    with connect_db() as connection:
        return {"customer_id": customer_id, "recommendations": _recommend(connection, customer_id)}


@app.get("/customer-segment/{customer_id}")
def customer_segment(customer_id: int):
    with connect_db() as connection:
        customer = _customer_or_404(connection, customer_id)
        return {"customer_id": customer_id, "segment": _segment(connection, customer)}


@app.get("/customer-dashboard/{customer_id}")
def customer_dashboard(customer_id: int, user: dict = Depends(_current_user)):
    _require_owner(customer_id, user)
    with connect_db() as connection:
        customer = dict(_customer_or_404(connection, customer_id))
        customer["email"] = customer["email"]
        predictions = []
        for product in _recommend(connection, customer_id):
            features = _features(connection, customer_id, product["product_id"])
            matrix = pd.DataFrame([{key: features[key] for key in PURCHASE_MODEL["features"]}]) if PURCHASE_MODEL else None
            probability = float(PURCHASE_MODEL["model"].predict_proba(matrix)[0, 1]) if matrix is not None else None
            predictions.append({"product_id": product["product_id"], "purchase_probability": round(probability, 4) if probability is not None else None})
        order_count = connection.execute("SELECT COUNT(*) FROM orders WHERE customer_id=? AND order_status!='cancelled'", (customer_id,)).fetchone()[0]
        return {
            "customer": customer, "total_orders": order_count, "total_spending": customer["total_spending"],
            "average_order_value": customer["average_order_value"], "previous_purchases": customer["total_purchases"],
            "customer_segment": _segment(connection, connection.execute("SELECT * FROM customers WHERE customer_id=?", (customer_id,)).fetchone()),
            "recommended_products": _recommend(connection, customer_id), "purchase_predictions": predictions,
        }


@app.get("/admin/sales-summary")
def sales_summary(_: dict = Depends(_admin)):
    with connect_db() as connection:
        summary = connection.execute("SELECT COUNT(*) AS total_orders, ROUND(COALESCE(SUM(total_amount),0),2) AS total_revenue, ROUND(COALESCE(AVG(total_amount),0),2) AS average_order_value FROM orders WHERE order_status!='cancelled'").fetchone()
        payments = [dict(row) for row in connection.execute("SELECT payment_method, COUNT(*) AS orders FROM orders WHERE order_status!='cancelled' GROUP BY payment_method ORDER BY orders DESC")]
        monthly = [dict(row) for row in connection.execute("SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(total_amount),2) AS revenue FROM orders WHERE order_status!='cancelled' GROUP BY month ORDER BY month")]
        return {**dict(summary), "payment_methods": payments, "monthly_revenue": monthly}


@app.get("/admin/top-products")
def top_products(_: dict = Depends(_admin), limit: int = Query(default=10, ge=1, le=50)):
    with connect_db() as connection:
        return [dict(row) for row in connection.execute("""SELECT p.product_id, p.product_name, p.category, SUM(oi.quantity) AS units_sold,
            ROUND(SUM(oi.quantity*oi.unit_price),2) AS revenue FROM products p JOIN order_items oi USING(product_id)
            JOIN orders o USING(order_id) WHERE o.order_status!='cancelled' GROUP BY p.product_id ORDER BY revenue DESC LIMIT ?""", (limit,))]


@app.get("/admin/top-customers")
def top_customers(_: dict = Depends(_admin), limit: int = Query(default=10, ge=1, le=50)):
    with connect_db() as connection:
        return [dict(row) for row in connection.execute("""SELECT c.customer_id, c.full_name, c.city, COUNT(o.order_id) AS total_orders,
            ROUND(SUM(o.total_amount),2) AS total_spending FROM customers c JOIN orders o USING(customer_id)
            WHERE o.order_status!='cancelled' GROUP BY c.customer_id ORDER BY total_spending DESC LIMIT ?""", (limit,))]


@app.get("/admin/category-sales")
def category_sales(_: dict = Depends(_admin)):
    with connect_db() as connection:
        return [dict(row) for row in connection.execute("""SELECT p.category, SUM(oi.quantity) AS units_sold,
            ROUND(SUM(oi.quantity*oi.unit_price),2) AS revenue FROM products p JOIN order_items oi USING(product_id)
            JOIN orders o USING(order_id) WHERE o.order_status!='cancelled' GROUP BY p.category ORDER BY revenue DESC""")]


@app.get("/admin/customer-segments")
def customer_segments(_: dict = Depends(_admin)):
    with connect_db() as connection:
        rows = connection.execute("SELECT * FROM customers").fetchall()
        counts = {}
        for customer in rows:
            label = _segment(connection, customer)
            counts[label] = counts.get(label, 0) + 1
        return {"segments": [{"segment": label, "customers": count} for label, count in sorted(counts.items())], "clusters": SEGMENT_MODEL["best_k"], "silhouette_scores": SEGMENT_MODEL["silhouette_scores"]}


@app.get("/admin/low-stock-products")
def low_stock_products(_: dict = Depends(_admin), threshold: int = Query(default=10, ge=0, le=100)):
    with connect_db() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM products WHERE stock <= ? ORDER BY stock, product_name", (threshold,))]


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")