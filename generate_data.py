"""
generate_data.py
Generates synthetic online retail dataset
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

N_CUSTOMERS = 300
N_PRODUCTS = 100
N_TRANS = 3000

CATEGORIES = {
    "Electronics": ["Wireless Headphones", "Bluetooth Speaker", "Smartwatch", "USB Hub",
                    "Laptop Stand", "Webcam", "Mechanical Keyboard", "Gaming Mouse",
                    "Power Bank", "LED Desk Lamp"],
    "Clothing": ["Cotton T-Shirt", "Denim Jeans", "Casual Hoodie", "Running Shorts",
                 "Winter Jacket", "Formal Shirt", "Yoga Pants", "Sports Socks",
                 "Baseball Cap", "Leather Belt"],
    "Books": ["Python for Data Science", "Machine Learning Basics", "Clean Code",
              "Atomic Habits", "The Pragmatic Programmer", "Deep Work",
              "System Design", "JavaScript: The Good Parts", "Data Structures",
              "The Lean Startup"],
    "Home & Kitchen": ["Non-Stick Pan", "Electric Kettle", "Coffee Maker", "Air Purifier",
                       "Vacuum Cleaner", "Cutting Board", "Knife Set", "Blender",
                       "Toaster", "Rice Cooker"],
    "Sports": ["Yoga Mat", "Resistance Bands", "Dumbbells Set", "Jump Rope",
               "Water Bottle", "Foam Roller", "Cycling Gloves", "Running Shoes",
               "Badminton Racket", "Skipping Rope"],
    "Beauty": ["Face Moisturizer", "Sunscreen SPF50", "Vitamin C Serum",
               "Shampoo", "Conditioner", "Face Wash", "Lip Balm",
               "Eye Cream", "Body Lotion", "Hair Oil"],
    "Toys": ["LEGO City Set", "Puzzle 1000 pcs", "Board Game", "RC Car",
             "Building Blocks", "Action Figure", "Doll House", "Card Game",
             "Art Kit", "Science Kit"],
    "Food": ["Organic Honey", "Green Tea", "Protein Powder", "Almonds 500g",
             "Dark Chocolate", "Olive Oil", "Quinoa", "Oat Meal",
             "Mixed Nuts", "Coffee Beans"],
    "Accessories": ["Leather Wallet", "Sunglasses", "Travel Bag", "Phone Case",
                    "Umbrella", "Key Ring", "Watch Strap", "Card Holder",
                    "Tote Bag", "Laptop Bag"],
    "Stationery": ["Notebook Set", "Ball Pens 10pk", "Sticky Notes", "Highlighters",
                   "Planner 2025", "Sketch Pens", "File Organizer", "Stapler",
                   "Whiteboard Markers", "Tape Dispenser"],
}

PRICE_RANGE = {
    "Electronics": (499, 4999), "Clothing": (199, 1999),
    "Books": (149, 799), "Home & Kitchen": (299, 3999),
    "Sports": (199, 2999), "Beauty": (99, 999),
    "Toys": (299, 2499), "Food": (99, 799),
    "Accessories": (199, 1999), "Stationery": (49, 499),
}

products = []
pid = 1
for cat, items in CATEGORIES.items():
    lo, hi = PRICE_RANGE[cat]
    for item in items:
        products.append({
            "product_id": f"P{str(pid).zfill(3)}",
            "product_name": item,
            "category": cat,
            "price": round(np.random.uniform(lo, hi), 2),
            "avg_rating": round(np.random.uniform(3.2, 5.0), 1),
            "stock": np.random.randint(10, 500),
        })
        pid += 1

products_df = pd.DataFrame(products)
products_df.to_csv("data/products.csv", index=False)

CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

PREF = {
    "18-24": [0.25, 0.20, 0.10, 0.05, 0.15, 0.10, 0.05, 0.05, 0.03, 0.02],
    "25-34": [0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.03, 0.07, 0.05, 0.05],
    "35-44": [0.15, 0.15, 0.15, 0.15, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04],
    "45-54": [0.10, 0.12, 0.15, 0.20, 0.08, 0.10, 0.10, 0.08, 0.04, 0.03],
    "55+":   [0.08, 0.10, 0.20, 0.22, 0.05, 0.08, 0.12, 0.08, 0.04, 0.03],
}

customers = []
for i in range(N_CUSTOMERS):
    age_g = np.random.choice(AGE_GROUPS, p=[0.20, 0.30, 0.25, 0.15, 0.10])
    customers.append({
        "customer_id": f"C{str(i+1).zfill(4)}",
        "age_group": age_g,
        "gender": np.random.choice(["Male", "Female", "Other"], p=[0.52, 0.45, 0.03]),
        "city": np.random.choice(CITIES),
        "member_since": np.random.randint(2018, 2025),
        "total_orders": np.random.randint(1, 80),
        "pref_category": np.random.choice(list(CATEGORIES.keys()), p=PREF[age_g]),
    })

customers_df = pd.DataFrame(customers)
customers_df.to_csv("data/customers.csv", index=False)

cat_list = list(CATEGORIES.keys())
cat_to_pids = {cat: products_df[products_df["category"] == cat]["product_id"].tolist()
               for cat in cat_list}

transactions = []
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
for i in range(N_TRANS):
    cust = customers_df.sample(1).iloc[0]
    pref = cust["pref_category"]
    age_g = cust["age_group"]
    if np.random.random() < 0.60:
        cat = pref
    else:
        cat = np.random.choice(cat_list, p=PREF[age_g])
    pid = np.random.choice(cat_to_pids[cat])
    price = products_df[products_df["product_id"] == pid]["price"].values[0]
    qty = np.random.randint(1, 4)
    transactions.append({
        "transaction_id": f"T{str(i+1).zfill(5)}",
        "customer_id": cust["customer_id"],
        "product_id": pid,
        "quantity": qty,
        "purchase_date": pd.Timestamp(np.random.choice(dates)).strftime("%Y-%m-%d"),
        "total_price": round(price * qty, 2),
    })

trans_df = pd.DataFrame(transactions)
trans_df.to_csv("data/transactions.csv", index=False)

bought_pairs = trans_df[["customer_id", "product_id"]].drop_duplicates()
rated = bought_pairs.sample(frac=0.60, random_state=42).copy()

def gen_rating(pid):
    avg = products_df[products_df["product_id"] == pid]["avg_rating"].values[0]
    r = int(np.clip(np.round(np.random.normal(avg, 0.7)), 1, 5))
    return r

rated["rating"] = rated["product_id"].apply(gen_rating)
rated.to_csv("data/ratings.csv", index=False)

print(f"✅ Products     : {len(products_df)}")
print(f"✅ Customers    : {len(customers_df)}")
print(f"✅ Transactions : {len(trans_df)}")
print(f"✅ Ratings      : {len(rated)}")
