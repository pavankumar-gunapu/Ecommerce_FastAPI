import pytest
from fastapi.testclient import TestClient

from smartcart.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTCART_DB", str(tmp_path / "smartcart-test.db"))
    with TestClient(app) as test_client:
        yield test_client


def test_customer_journey_and_admin_access(client):
    products = client.get("/products?limit=50")
    assert products.status_code == 200 and len(products.json()) == 50
    product = next(item for item in products.json() if item["stock"] > 1)
    assert client.get(f"/products/{product['product_id']}").status_code == 200
    assert client.get("/products?search=Wireless").status_code == 200

    customer = client.post("/register", json={"full_name": "Taylor Reed", "email": "taylor@example.com", "password": "CartDemo123!", "age": 29, "city": "Austin"})
    assert customer.status_code == 201
    customer_id = customer.json()["customer"]["customer_id"]
    token = customer.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/register", json={"full_name": "Taylor Reed", "email": "taylor@example.com", "password": "CartDemo123!", "age": 29, "city": "Austin"}).status_code == 409
    assert client.post("/login", json={"email": "taylor@example.com", "password": "wrong-password"}).status_code == 401
    assert client.post("/login", json={"email": "taylor@example.com", "password": "CartDemo123!"}).status_code == 200

    created_customer = client.post("/customers", json={"age": 32, "city": "Denver"})
    assert created_customer.status_code == 201
    assert client.get(f"/customers/{created_customer.json()['customer_id']}").status_code == 200
    assert client.put(f"/customers/{created_customer.json()['customer_id']}", json={"city": "Boston"}).json()["city"] == "Boston"
    assert client.get("/customers").status_code == 200
    assert client.delete(f"/customers/{created_customer.json()['customer_id']}").status_code == 204

    new_product = client.post("/products", json={"product_name": "Test Lamp", "category": "Home", "price": 25, "rating": 4.2, "stock": 8})
    assert new_product.status_code == 201
    changed_product = client.put(f"/products/{new_product.json()['product_id']}", json={"product_name": "Test Lamp", "category": "Home", "price": 26, "rating": 4.4, "stock": 8})
    assert changed_product.json()["price"] == 26
    assert client.delete(f"/products/{new_product.json()['product_id']}").status_code == 204

    assert client.post("/cart", json={"customer_id": customer_id, "product_id": product["product_id"], "quantity": 1}, headers=headers).status_code == 201
    cart_item = client.get(f"/cart/{customer_id}", headers=headers).json()[0]
    assert client.put(f"/cart/{cart_item['cart_id']}", json={"quantity": 2}, headers=headers).status_code == 200
    second_product = next(item for item in products.json() if item["product_id"] != product["product_id"] and item["stock"] > 0)
    removable = client.post("/cart", json={"customer_id": customer_id, "product_id": second_product["product_id"], "quantity": 1}, headers=headers).json()
    assert client.delete(f"/cart/{removable['cart_id']}", headers=headers).status_code == 204

    order = client.post("/orders", json={"payment_method": "Card"}, headers=headers)
    assert order.status_code == 201
    order_id = order.json()["order_id"]
    assert client.get("/orders", headers=headers).status_code == 200
    assert client.get(f"/orders/{order_id}", headers=headers).status_code == 200
    assert client.get(f"/customers/{customer_id}/orders", headers=headers).status_code == 200
    prediction = client.post("/predict-purchase", json={"customer_id": customer_id, "product_id": product["product_id"]})
    assert prediction.status_code == 200 and 0 <= prediction.json()["purchase_probability"] <= 1
    assert client.get(f"/recommendations/{customer_id}").status_code == 200
    assert client.get(f"/customer-segment/{customer_id}").status_code == 200
    dashboard = client.get(f"/customer-dashboard/{customer_id}", headers=headers)
    assert dashboard.status_code == 200 and dashboard.json()["total_orders"] == 1
    assert client.get("/admin/sales-summary", headers=headers).status_code == 403

    admin = client.post("/login", json={"email": "admin@smartcart.com", "password": "Admin123!"})
    admin_headers = {"Authorization": f"Bearer {admin.json()['access_token']}"}
    for path in ("sales-summary", "top-products", "top-customers", "category-sales", "customer-segments", "low-stock-products"):
        assert client.get(f"/admin/{path}", headers=admin_headers).status_code == 200


def test_validation_and_expected_errors(client):
    assert client.post("/register", json={"full_name": "A", "email": "not-an-email", "password": "short", "age": 17, "city": "Austin"}).status_code == 422
    assert client.get("/products/99999").status_code == 404
    assert client.post("/predict-purchase", json={"customer_id": 99999, "product_id": 1}).status_code == 404
    assert client.post("/login", json={"email": "missing@example.com", "password": "Unknown123!"}).status_code == 401
    assert client.get("/admin/sales-summary").status_code == 401