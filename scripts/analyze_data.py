"""Summarize the generated storefront data and save a few EDA charts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from smartcart.data import DATA_DIR

customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
orders["revenue"] = orders["quantity"] * orders["price"]
valid = orders[orders.order_status != "cancelled"]
sales = valid.merge(products[["product_id", "product_name", "category"]], on="product_id")
sales = sales.merge(customers[["customer_id", "city"]], on="customer_id")
output = DATA_DIR / "eda"
output.mkdir(parents=True, exist_ok=True)

category_sales = sales.groupby("category").revenue.sum().sort_values(ascending=False)
product_sales = sales.groupby("product_name").revenue.sum().sort_values(ascending=False)
customer_sales = sales.groupby("customer_id").revenue.sum().sort_values(ascending=False)
repeat_pct = (valid.groupby("customer_id").order_id.nunique().gt(1).mean() * 100)
print(f"Highest-sales category: {category_sales.index[0]} (${category_sales.iloc[0]:,.2f})")
print("Top revenue products:\n", product_sales.head(5).to_string())
print("Top-spending customers:\n", customer_sales.head(5).to_string())
print(f"Average order value: ${valid.revenue.mean():,.2f}")
print(f"Most-used payment method: {valid.payment_method.value_counts().index[0]}")
print(f"City with most customers: {customers.city.value_counts().index[0]}")
print("Highest product ratings:\n", products.nlargest(5, "rating")[["product_name", "rating"]].to_string(index=False))
print(f"Customers with repeat purchases: {repeat_pct:.1f}%")
print("Monthly revenue:\n", valid.assign(month=valid.order_date.dt.to_period("M")).groupby("month").revenue.sum().to_string())
print("Low-stock products (10 or fewer):\n", products[products.stock <= 10][["product_name", "stock"]].to_string(index=False))

for series, title, filename, ylabel in [
    (category_sales, "Revenue by category", "category-revenue.png", "Revenue ($)"),
    (valid.assign(month=valid.order_date.dt.to_period("M")).groupby("month").revenue.sum(), "Monthly revenue", "monthly-revenue.png", "Revenue ($)"),
    (products.nlargest(10, "rating").set_index("product_name").rating, "Top product ratings", "product-ratings.png", "Rating"),
]:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    series.plot(kind="bar", ax=axis, color="#477451")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.set_xlabel("")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output / filename, dpi=150)
    plt.close(figure)
print(f"Charts saved to {output}")