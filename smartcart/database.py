"""SQLite schema, connection, and synthetic-data seeding."""

import os
import sqlite3
from pathlib import Path

import pandas as pd

from smartcart.data import DATA_DIR, generate_datasets

DB_PATH = Path(os.getenv("SMARTCART_DB", Path(__file__).resolve().parent.parent / "data" / "smartcart.db"))

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL DEFAULT 'SmartCart Customer',
    email TEXT UNIQUE COLLATE NOCASE,
    age INTEGER NOT NULL CHECK(age BETWEEN 18 AND 100),
    gender TEXT NOT NULL DEFAULT 'Not specified',
    city TEXT NOT NULL DEFAULT 'Unknown',
    registration_date TEXT NOT NULL,
    total_purchases INTEGER NOT NULL DEFAULT 0 CHECK(total_purchases >= 0),
    total_spending REAL NOT NULL DEFAULT 0 CHECK(total_spending >= 0),
    average_order_value REAL NOT NULL DEFAULT 0 CHECK(average_order_value >= 0),
    website_visits INTEGER NOT NULL DEFAULT 0 CHECK(website_visits >= 0)
);
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER UNIQUE REFERENCES customers(customer_id) ON DELETE CASCADE,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('customer', 'admin')) DEFAULT 'customer',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL CHECK(price > 0),
    rating REAL NOT NULL DEFAULT 0 CHECK(rating BETWEEN 0 AND 5),
    stock INTEGER NOT NULL DEFAULT 0 CHECK(stock >= 0),
    discount REAL NOT NULL DEFAULT 0 CHECK(discount BETWEEN 0 AND 100)
);
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    order_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    payment_method TEXT NOT NULL,
    order_status TEXT NOT NULL DEFAULT 'processing',
    total_amount REAL NOT NULL DEFAULT 0 CHECK(total_amount >= 0)
);
CREATE TABLE IF NOT EXISTS order_items (
    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE RESTRICT,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price REAL NOT NULL CHECK(unit_price > 0)
);
CREATE TABLE IF NOT EXISTS cart (
    cart_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    UNIQUE(customer_id, product_id)
);
CREATE TABLE IF NOT EXISTS wishlist (
    wishlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id, product_id)
);
CREATE TABLE IF NOT EXISTS customer_activity (
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    product_views INTEGER NOT NULL DEFAULT 0 CHECK(product_views >= 0),
    cart_additions INTEGER NOT NULL DEFAULT 0 CHECK(cart_additions >= 0),
    wishlist_additions INTEGER NOT NULL DEFAULT 0 CHECK(wishlist_additions >= 0),
    previous_purchases INTEGER NOT NULL DEFAULT 0 CHECK(previous_purchases >= 0),
    time_spent REAL NOT NULL DEFAULT 0 CHECK(time_spent >= 0),
    purchased INTEGER NOT NULL DEFAULT 0 CHECK(purchased IN (0, 1)),
    PRIMARY KEY(customer_id, product_id)
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);
"""


def connect_db() -> sqlite3.Connection:
    path = Path(os.getenv("SMARTCART_DB", DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def init_db() -> None:
    with connect_db() as connection:
        connection.executescript(SCHEMA)


def seed_database() -> None:
    if not (DATA_DIR / "customers.csv").exists():
        generate_datasets()
    with connect_db() as connection:
        if connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0]:
            return
        customers = pd.read_csv(DATA_DIR / "customers.csv").to_dict(orient="records")
        for row in customers:
            connection.execute(
                """INSERT INTO customers(customer_id, age, gender, city, registration_date,
                   total_purchases, total_spending, average_order_value, website_visits)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row[key] for key in ("customer_id", "age", "gender", "city", "registration_date", "total_purchases", "total_spending", "average_order_value", "website_visits")),
            )
        products = pd.read_csv(DATA_DIR / "products.csv").to_dict(orient="records")
        connection.executemany(
            "INSERT INTO products(product_id, product_name, category, price, rating, stock, discount) VALUES (:product_id, :product_name, :category, :price, :rating, :stock, :discount)",
            products,
        )
        orders = pd.read_csv(DATA_DIR / "orders.csv").to_dict(orient="records")
        for row in orders:
            total = float(row["price"]) * int(row["quantity"])
            connection.execute(
                "INSERT INTO orders(order_id, customer_id, order_date, payment_method, order_status, total_amount) VALUES (?, ?, ?, ?, ?, ?)",
                (int(row["order_id"]), int(row["customer_id"]), row["order_date"], row["payment_method"], row["order_status"], total),
            )
            connection.execute(
                "INSERT INTO order_items(order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
                (int(row["order_id"]), int(row["product_id"]), int(row["quantity"]), float(row["price"])),
            )
        activity = pd.read_csv(DATA_DIR / "customer_activity.csv").to_dict(orient="records")
        connection.executemany(
            """INSERT INTO customer_activity(customer_id, product_id, product_views, cart_additions,
               wishlist_additions, previous_purchases, time_spent, purchased)
               VALUES (:customer_id, :product_id, :product_views, :cart_additions,
               :wishlist_additions, :previous_purchases, :time_spent, :purchased)""",
            activity,
        )