"""Generate reproducible CSV datasets for the SmartCart demo."""

from pathlib import Path
import random

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CATEGORIES = ["Electronics", "Home", "Books", "Beauty", "Sports"]
CITIES = ["Austin", "Boston", "Chicago", "Denver", "Seattle", "Miami"]


def generate_datasets(output_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    rng = random.Random(2026)
    np_rng = np.random.default_rng(2026)
    output_dir.mkdir(parents=True, exist_ok=True)

    products = pd.DataFrame([
        {
            "product_id": product_id,
            "product_name": f"{['Wireless Headphones', 'Smart Watch', 'Everyday', 'Premium'][product_id % 4]} {CATEGORIES[(product_id - 1) % len(CATEGORIES)]} {product_id:02d}",
            "category": CATEGORIES[(product_id - 1) % len(CATEGORIES)],
            "price": round(rng.uniform(12, 420), 2),
            "rating": round(rng.uniform(2.8, 5.0), 1),
            "stock": rng.randint(0, 90),
            "discount": rng.choice([0, 0, 5, 10, 15, 20]),
        }
        for product_id in range(1, 51)
    ])
    customer_seed = [{
        "customer_id": customer_id,
        "age": rng.randint(18, 75),
        "gender": rng.choice(["Female", "Male", "Non-binary"]),
        "city": rng.choice(CITIES),
        "registration_date": (pd.Timestamp("2026-01-01") - pd.Timedelta(days=rng.randint(30, 1500))).date().isoformat(),
        "website_visits": rng.randint(5, 180),
    } for customer_id in range(1, 101)]
    order_rows = []
    for order_id in range(1, 501):
        customer = rng.choice(customer_seed)
        product = products.iloc[rng.randrange(len(products))]
        quantity = rng.randint(1, 4)
        order_rows.append({
            "order_id": order_id,
            "customer_id": customer["customer_id"],
            "product_id": int(product.product_id),
            "quantity": quantity,
            "order_date": (pd.Timestamp("2026-09-30") - pd.Timedelta(days=rng.randint(0, 540))).date().isoformat(),
            "price": round(float(product.price) * (1 - float(product.discount) / 100), 2),
            "payment_method": rng.choice(["Card", "PayPal", "Apple Pay", "Gift Card"]),
            "order_status": rng.choices(["delivered", "processing", "cancelled"], [0.78, 0.16, 0.06])[0],
        })
    orders = pd.DataFrame(order_rows)
    paid_orders = orders[orders.order_status != "cancelled"].assign(line_total=lambda frame: frame.quantity * frame.price)
    totals = paid_orders.groupby("customer_id").agg(
        total_purchases=("order_id", "nunique"), total_spending=("line_total", "sum")
    )
    customers = pd.DataFrame(customer_seed).merge(totals, how="left", left_on="customer_id", right_index=True)
    customers[["total_purchases", "total_spending"]] = customers[["total_purchases", "total_spending"]].fillna(0)
    customers["total_purchases"] = customers.total_purchases.astype(int)
    customers["total_spending"] = customers.total_spending.round(2)
    customers["average_order_value"] = np.where(
        customers.total_purchases > 0,
        (customers.total_spending / customers.total_purchases).round(2), 0,
    )
    customers = customers[["customer_id", "age", "gender", "city", "registration_date", "total_purchases", "total_spending", "average_order_value", "website_visits"]]

    pair_orders = orders[orders.order_status != "cancelled"].groupby(["customer_id", "product_id"]).quantity.sum()
    activities = []
    for customer in customers.itertuples(index=False):
        for product in products.itertuples(index=False):
            prior = int(pair_orders.get((customer.customer_id, product.product_id), 0))
            views = rng.randint(0, 18)
            cart_additions = rng.randint(0, min(4, views + 1))
            wishlists = rng.randint(0, 3)
            time_spent = round(rng.uniform(5, 600), 1)
            logit = -2.3 + 0.20 * views + 0.8 * cart_additions + 0.38 * prior + 0.3 * wishlists + (float(product.rating) - 3.5) * 0.35
            probability = 1 / (1 + np.exp(-logit))
            activities.append({
                "customer_id": customer.customer_id,
                "product_id": product.product_id,
                "product_views": views,
                "cart_additions": cart_additions,
                "wishlist_additions": wishlists,
                "previous_purchases": prior,
                "time_spent": time_spent,
                "purchased": int(np_rng.random() < probability),
            })
    activity = pd.DataFrame(activities)
    datasets = {"customers": customers, "products": products, "orders": orders, "customer_activity": activity}
    for name, frame in datasets.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return datasets


if __name__ == "__main__":
    frames = generate_datasets()
    for name, frame in frames.items():
        print(f"{name}: {len(frame)} records")