# dataset_generator_scaled.py
# Scaled-down version for development/submission
# Full scale: NUM_SESSIONS=2,000,000 / NUM_TRANSACTIONS=500,000 (~3-5GB)
# This scale: NUM_SESSIONS=50,000 / NUM_TRANSACTIONS=10,000 (~100-150MB)
# Increase multipliers to scale up on 16GB+ RAM machines

import json
import random
import datetime
import uuid
import threading
import numpy as np
from faker import Faker

fake = Faker()

# --- Configuration (Scaled for development; comment out and use originals for full run) ---
NUM_USERS = 1000          # Original: 10,000
NUM_PRODUCTS = 500        # Original: 5,000
NUM_CATEGORIES = 25       # Same
NUM_TRANSACTIONS = 10000  # Original: 500,000
NUM_SESSIONS = 50000      # Original: 2,000,000
TIMESPAN_DAYS = 90
MAX_ITERATIONS = (NUM_SESSIONS + NUM_TRANSACTIONS) * 2

# --- Reproducibility ---
np.random.seed(42)
random.seed(42)
Faker.seed(42)

print("Initializing dataset generation (scaled mode)...")
print(f"Target: {NUM_SESSIONS:,} sessions, {NUM_TRANSACTIONS:,} transactions")

# --- ID Generators ---
def generate_session_id():
    return f"sess_{uuid.uuid4().hex[:10]}"

def generate_transaction_id():
    return f"txn_{uuid.uuid4().hex[:12]}"

# --- Inventory Manager (thread-safe) ---
class InventoryManager:
    def __init__(self, products):
        self.products = {p["product_id"]: p for p in products}
        self.lock = threading.RLock()

    def update_stock(self, product_id, quantity):
        with self.lock:
            if product_id not in self.products:
                return False
            if self.products[product_id]["current_stock"] >= quantity:
                self.products[product_id]["current_stock"] -= quantity
                return True
            return False

    def get_product(self, product_id):
        with self.lock:
            return self.products.get(product_id)

# --- Page flow logic ---
def determine_page_type(position, previous_pages):
    """
    Models realistic e-commerce navigation patterns.
    Users typically start broad (home/search) and funnel toward purchase.
    """
    if position == 0:
        return random.choice(["home", "search", "category_listing"])

    if not previous_pages:
        return "home"

    prev_page = previous_pages[-1]["page_type"]

    transitions = {
        "home":             (["category_listing", "search", "product_detail"],        [0.5, 0.3, 0.2]),
        "category_listing": (["product_detail", "category_listing", "search", "home"],[0.7, 0.1, 0.1, 0.1]),
        "search":           (["product_detail", "search", "category_listing", "home"],[0.6, 0.2, 0.1, 0.1]),
        "product_detail":   (["product_detail", "cart", "category_listing", "search", "home"], [0.3, 0.3, 0.2, 0.1, 0.1]),
        "cart":             (["checkout", "product_detail", "category_listing", "home"],[0.6, 0.2, 0.1, 0.1]),
        "checkout":         (["confirmation", "cart", "home"],                          [0.8, 0.1, 0.1]),
        "confirmation":     (["home", "product_detail", "category_listing"],            [0.6, 0.2, 0.2]),
    }

    pages, weights = transitions.get(prev_page, (["home"], [1.0]))
    return random.choices(pages, weights=weights)[0]

def get_page_content(page_type, products_list, categories_list, inventory):
    """Returns appropriate product/category context for a given page type."""
    if page_type == "product_detail":
        for _ in range(10):
            product = random.choice(products_list)
            if product["is_active"] and product["current_stock"] > 0:
                category = next((c for c in categories_list if c["category_id"] == product["category_id"]), None)
                return product, category
        product = random.choice(products_list)
        category = next((c for c in categories_list if c["category_id"] == product["category_id"]), None)
        return product, category
    elif page_type == "category_listing":
        return None, random.choice(categories_list)
    return None, None

# ============================================================
# CATEGORY GENERATION
# ============================================================
categories = []
for cat_id in range(NUM_CATEGORIES):
    category = {
        "category_id": f"cat_{cat_id:03d}",
        "name": fake.company(),
        "subcategories": []
    }
    for sub_id in range(random.randint(3, 5)):
        category["subcategories"].append({
            "subcategory_id": f"sub_{cat_id:03d}_{sub_id:02d}",
            "name": fake.bs(),
            "profit_margin": round(random.uniform(0.1, 0.4), 2)
        })
    categories.append(category)

print(f"  ✓ Generated {len(categories)} categories")

# ============================================================
# PRODUCT GENERATION
# ============================================================
products = []
product_creation_start = datetime.datetime.now() - datetime.timedelta(days=TIMESPAN_DAYS * 2)

for prod_id in range(NUM_PRODUCTS):
    category = random.choice(categories)
    subcategory = random.choice(category["subcategories"])
    base_price = round(random.uniform(5, 500), 2)
    price_history = []

    initial_date = fake.date_time_between(
        start_date=product_creation_start,
        end_date=product_creation_start + datetime.timedelta(days=TIMESPAN_DAYS // 3)
    )
    price_history.append({"price": base_price, "date": initial_date.isoformat()})

    for _ in range(random.randint(0, 2)):
        change_date = fake.date_time_between(start_date=initial_date, end_date="now")
        new_price = round(base_price * random.uniform(0.8, 1.2), 2)
        price_history.append({"price": new_price, "date": change_date.isoformat()})
        initial_date = change_date

    price_history.sort(key=lambda x: x["date"])
    current_price = price_history[-1]["price"]

    products.append({
        "product_id": f"prod_{prod_id:05d}",
        "name": fake.catch_phrase().title(),
        "category_id": category["category_id"],
        "subcategory_id": subcategory["subcategory_id"],
        "base_price": current_price,
        "current_stock": random.randint(10, 1000),
        "is_active": random.choices([True, False], weights=[0.95, 0.05])[0],
        "price_history": price_history,
        "creation_date": price_history[0]["date"]
    })

print(f"  ✓ Generated {len(products)} products")

# ============================================================
# USER GENERATION
# ============================================================
users = []
for user_id in range(NUM_USERS):
    reg_date = fake.date_time_between(
        start_date=f"-{TIMESPAN_DAYS * 3}d",
        end_date=f"-{TIMESPAN_DAYS}d"
    )
    users.append({
        "user_id": f"user_{user_id:06d}",
        "geo_data": {
            "city": fake.city(),
            "state": fake.state_abbr(),
            "country": fake.country_code()
        },
        "registration_date": reg_date.isoformat(),
        "last_active": fake.date_time_between(start_date=reg_date, end_date="now").isoformat()
    })

print(f"  ✓ Generated {len(users)} users")

# ============================================================
# SESSION & TRANSACTION GENERATION
# ============================================================
inventory = InventoryManager(products)
sessions = []
transactions = []
transaction_counter = 0
session_counter = 0
iteration = 0

print("Generating sessions and transactions (this takes a few minutes)...")

while (session_counter < NUM_SESSIONS or transaction_counter < NUM_TRANSACTIONS) and iteration < MAX_ITERATIONS:
    iteration += 1

    # --- Session block ---
    if session_counter < NUM_SESSIONS:
        user = random.choice(users)
        session_id = generate_session_id()
        session_start = fake.date_time_between(start_date=f"-{TIMESPAN_DAYS}d", end_date="now")
        session_duration = random.randint(30, 3600)

        page_views = []
        viewed_products = set()
        cart_contents = {}

        time_slots = sorted(
            [0] + [random.randint(1, session_duration - 1) for _ in range(random.randint(3, 15))] + [session_duration]
        )

        for i in range(len(time_slots) - 1):
            view_duration = time_slots[i + 1] - time_slots[i]
            page_type = determine_page_type(i, page_views)
            product, category = get_page_content(page_type, products, categories, inventory)

            if page_type == "product_detail" and product:
                product_id = product["product_id"]
                viewed_products.add(product_id)
                if random.random() < 0.3:
                    if product_id not in cart_contents:
                        cart_contents[product_id] = {"quantity": 0, "price": product["base_price"]}
                    inv_product = inventory.get_product(product_id)
                    if inv_product:
                        max_possible = min(3, inv_product["current_stock"] - cart_contents[product_id]["quantity"])
                        if max_possible > 0:
                            cart_contents[product_id]["quantity"] += random.randint(1, max_possible)

            page_views.append({
                "timestamp": (session_start + datetime.timedelta(seconds=time_slots[i])).isoformat(),
                "page_type": page_type,
                "product_id": product["product_id"] if product else None,
                "category_id": category["category_id"] if category else None,
                "view_duration": view_duration
            })

        converted = False
        if cart_contents and any(p["page_type"] in ["checkout", "confirmation"] for p in page_views):
            converted = random.random() < 0.7

        session_geo = user["geo_data"].copy()
        session_geo["ip_address"] = fake.ipv4()

        sessions.append({
            "session_id": session_id,
            "user_id": user["user_id"],
            "start_time": session_start.isoformat(),
            "end_time": (session_start + datetime.timedelta(seconds=session_duration)).isoformat(),
            "duration_seconds": session_duration,
            "geo_data": session_geo,
            "device_profile": {
                "type": random.choice(["mobile", "desktop", "tablet"]),
                "os": random.choice(["iOS", "Android", "Windows", "macOS"]),
                "browser": random.choice(["Chrome", "Safari", "Firefox", "Edge"])
            },
            "viewed_products": list(viewed_products),
            "page_views": page_views,
            "cart_contents": {k: v for k, v in cart_contents.items() if v["quantity"] > 0},
            "conversion_status": "converted" if converted else "abandoned" if cart_contents else "browsed",
            "referrer": random.choice(["direct", "email", "social", "search_engine", "affiliate"])
        })
        session_counter += 1

        if converted and transaction_counter < NUM_TRANSACTIONS:
            transaction_items = []
            valid = True
            for prod_id, details in cart_contents.items():
                qty = details["quantity"]
                if qty > 0:
                    if inventory.update_stock(prod_id, qty):
                        transaction_items.append({
                            "product_id": prod_id,
                            "quantity": qty,
                            "unit_price": details["price"],
                            "subtotal": round(qty * details["price"], 2)
                        })
                    else:
                        valid = False
                        break
            if valid and transaction_items:
                subtotal = sum(item["subtotal"] for item in transaction_items)
                discount = 0
                if random.random() < 0.2:
                    discount = round(subtotal * random.choice([0.05, 0.1, 0.15, 0.2]), 2)
                transactions.append({
                    "transaction_id": generate_transaction_id(),
                    "session_id": session_id,
                    "user_id": user["user_id"],
                    "timestamp": (session_start + datetime.timedelta(seconds=session_duration)).isoformat(),
                    "items": transaction_items,
                    "subtotal": subtotal,
                    "discount": discount,
                    "total": round(subtotal - discount, 2),
                    "payment_method": random.choice(["credit_card", "paypal", "apple_pay", "crypto"]),
                    "status": "completed"
                })
                transaction_counter += 1

    # --- Standalone transaction block ---
    if transaction_counter < NUM_TRANSACTIONS and random.random() < 0.2:
        user = random.choice(users)
        sampled = random.sample(products, k=min(3, len(products)))
        transaction_items = []
        for product in sampled:
            if product["is_active"]:
                qty = random.randint(1, 3)
                if inventory.update_stock(product["product_id"], qty):
                    transaction_items.append({
                        "product_id": product["product_id"],
                        "quantity": qty,
                        "unit_price": product["base_price"],
                        "subtotal": round(qty * product["base_price"], 2)
                    })
        if transaction_items:
            subtotal = sum(item["subtotal"] for item in transaction_items)
            discount = round(subtotal * random.choice([0.05, 0.1, 0.15, 0.2]), 2) if random.random() < 0.2 else 0
            transactions.append({
                "transaction_id": generate_transaction_id(),
                "session_id": None,
                "user_id": user["user_id"],
                "timestamp": fake.date_time_between(start_date=f"-{TIMESPAN_DAYS}d", end_date="now").isoformat(),
                "items": transaction_items,
                "subtotal": subtotal,
                "discount": discount,
                "total": round(subtotal - discount, 2),
                "payment_method": random.choice(["credit_card", "paypal", "bank_transfer", "gift_card"]),
                "status": random.choice(["completed", "processing", "shipped", "delivered"])
            })
            transaction_counter += 1

    if iteration % 5000 == 0:
        print(f"  Progress: {session_counter:,}/{NUM_SESSIONS:,} sessions | "
              f"{transaction_counter:,}/{NUM_TRANSACTIONS:,} transactions")

# ============================================================
# EXPORT
# ============================================================
def json_serializer(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

import os
output_dir = os.path.dirname(os.path.abspath(__file__))
print("\nSaving datasets...")

with open(os.path.join(output_dir, "users.json"), "w") as f:
    json.dump(users, f, default=json_serializer)

with open(os.path.join(output_dir, "products.json"), "w") as f:
    json.dump(list(inventory.products.values()), f, default=json_serializer)

with open(os.path.join(output_dir, "categories.json"), "w") as f:
    json.dump(categories, f, default=json_serializer)

with open(os.path.join(output_dir, "transactions.json"), "w") as f:
    json.dump(transactions, f, default=json_serializer)

CHUNK_SIZE = 10000
for i in range(0, len(sessions), CHUNK_SIZE):
    chunk = sessions[i:i + CHUNK_SIZE]
    with open(os.path.join(output_dir, f"sessions_{i // CHUNK_SIZE}.json"), "w") as f:
        json.dump(chunk, f, default=json_serializer)

print(f"""
✓ Dataset generation complete!
  Sessions    : {len(sessions):,} (target: {NUM_SESSIONS:,})
  Transactions: {len(transactions):,} (target: {NUM_TRANSACTIONS:,})
  Remaining stock total: {sum(p['current_stock'] for p in inventory.products.values()):,}
  
Files saved to: {output_dir}
""")
