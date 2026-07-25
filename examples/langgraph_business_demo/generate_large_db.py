"""
Generates a 1,000-order sample database for large-scale enterprise stress testing of Chronos-Agent.
"""

import json
import random
from pathlib import Path

CUSTOMERS = [
    "Alice Smith", "Bob Jones", "Carol Danvers", "David Miller", "Emma Watson",
    "Frank Castle", "Grace Hopper", "Henry Ford", "Irene Adler", "Jack Sparrow",
    "Katherine Johnson", "Liam Neeson", "Maya Lin", "Nathan Drake", "Olivia Wilde",
    "Peter Parker", "Quinn Fabray", "Rachel Green", "Steve Rogers", "Tony Stark"
]

ITEMS = [
    ("Wireless Headphones", 120.00),
    ("USB-C Fast Charger", 25.00),
    ("Ergonomic Keyboard", 85.00),
    ("4K Monitor 27-inch", 350.00),
    ("Bluetooth Speaker", 45.00),
    ("Mechanical Gaming Mouse", 60.00),
    ("Laptop Stand Aluminum", 35.00),
    ("HD Webcam 1080p", 55.00),
    ("Noise Canceling Earbuds", 150.00),
    ("Smartwatch Fitness Tracker", 199.00)
]

def generate_1000_orders(output_dir: str):
    db_path = Path(output_dir)
    db_path.mkdir(parents=True, exist_ok=True)

    orders = {}
    random.seed(42)  # Deterministic seed for reproducible testing

    for order_id_num in range(1000, 2000):
        order_id = str(order_id_num)
        customer = random.choice(CUSTOMERS)
        item_name, item_price = random.choice(ITEMS)
        qty = random.randint(1, 3)
        total_amount = round(item_price * qty, 2)

        orders[order_id] = {
            "order_id": order_id,
            "customer": customer,
            "items": [item_name] * qty,
            "total_amount": total_amount,
            "status": "COMPLETED"
        }

    orders_file = db_path / "orders.json"
    orders_file.write_text(json.dumps(orders, indent=2))

    ledger = {
        "company_balance": 1000000.00,  # $1,000,000 starting ledger
        "total_refunds_processed": 0.00,
        "refund_history": []
    }
    ledger_file = db_path / "financial_ledger.json"
    ledger_file.write_text(json.dumps(ledger, indent=2))

    print(f"✅ Generated 1,000 orders in '{orders_file}' ({orders_file.stat().st_size / 1024:.1f} KB)")
    print(f"✅ Seeding initial financial ledger in '{ledger_file}' ($1,000,000.00 balance)")

if __name__ == "__main__":
    target = Path(__file__).parent / "mock_db"
    generate_1000_orders(str(target))
