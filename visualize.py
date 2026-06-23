"""
visualize.py
Generates all EDA charts for the online retail recommendation dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

products_df = pd.read_csv("data/products.csv")
customers_df = pd.read_csv("data/customers.csv")
trans_df = pd.read_csv("data/transactions.csv")
ratings_df = pd.read_csv("data/ratings.csv")

# Merge transactions with product info
trans_full = trans_df.merge(products_df[["product_id", "category", "price"]], on="product_id")
trans_full["purchase_date"] = pd.to_datetime(trans_full["purchase_date"])
trans_full["month"] = trans_full["purchase_date"].dt.to_period("M").astype(str)

CAT_COLORS = {
    "Electronics": "#3498db", "Clothing": "#e74c3c", "Books": "#f39c12",
    "Home & Kitchen": "#2ecc71", "Sports": "#9b59b6", "Beauty": "#1abc9c",
    "Toys": "#e67e22", "Food": "#27ae60", "Accessories": "#8e44ad",
    "Stationery": "#2980b9"
}
COLOR_LIST = list(CAT_COLORS.values())

print("📊 Generating visualizations...")

# ── 01: Sales by Category (Bar) ──────────────────────────────────────────────
cat_sales = trans_full.groupby("category")["total_price"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
colors = [CAT_COLORS.get(c, "#95a5a6") for c in cat_sales.index]
bars = ax.bar(cat_sales.index, cat_sales.values / 1e5, color=colors,
              edgecolor="white", linewidth=1, width=0.6)
for bar, val in zip(bars, cat_sales.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"₹{val/1e5:.1f}L", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("Total Sales by Category", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Category", fontsize=12)
ax.set_ylabel("Revenue (₹ Lakhs)", fontsize=12)
ax.set_xticklabels(cat_sales.index, rotation=30, ha="right")
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/01_sales_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 01_sales_by_category.png")

# ── 02: Top 15 Most Purchased Products ───────────────────────────────────────
top_products = (trans_df.groupby("product_id")["quantity"]
                .sum()
                .reset_index()
                .merge(products_df[["product_id", "product_name", "category"]], on="product_id")
                .sort_values("quantity", ascending=False)
                .head(15))

fig, ax = plt.subplots(figsize=(10, 7))
colors_p = [CAT_COLORS.get(c, "#95a5a6") for c in top_products["category"]]
ax.barh(top_products["product_name"], top_products["quantity"],
        color=colors_p, edgecolor="white", linewidth=0.8, height=0.65)
ax.set_title("Top 15 Most Purchased Products", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Total Units Sold", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="x", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
# Category legend
from matplotlib.patches import Patch
cats_shown = top_products["category"].unique()
handles = [Patch(color=CAT_COLORS.get(c, "#95a5a6"), label=c) for c in cats_shown]
ax.legend(handles=handles, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/02_top_products.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 02_top_products.png")

# ── 03: Monthly Revenue Trend ────────────────────────────────────────────────
monthly_rev = trans_full.groupby("month")["total_price"].sum().reset_index()
monthly_rev = monthly_rev.sort_values("month")
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(monthly_rev["month"], monthly_rev["total_price"] / 1e5,
        color="#3498db", linewidth=2.2, marker="o", markersize=5)
ax.fill_between(range(len(monthly_rev)), monthly_rev["total_price"] / 1e5, alpha=0.12, color="#3498db")
ax.set_xticks(range(len(monthly_rev)))
ax.set_xticklabels(monthly_rev["month"], rotation=45, ha="right", fontsize=8)
ax.set_title("Monthly Revenue Trend (2023–2024)", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Revenue (₹ Lakhs)", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(alpha=0.3, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/03_monthly_revenue.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 03_monthly_revenue.png")

# ── 04: Rating Distribution ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
rating_counts = ratings_df["rating"].value_counts().sort_index()
colors_r = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#27ae60"]
bars = ax.bar(rating_counts.index, rating_counts.values, color=colors_r,
              edgecolor="white", linewidth=1.2, width=0.6)
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Customer Rating Distribution", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Rating (1–5 Stars)", fontsize=12)
ax.set_ylabel("Number of Ratings", fontsize=12)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/04_rating_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 04_rating_distribution.png")

# ── 05: Avg Rating by Category ────────────────────────────────────────────────
rat_cat = (ratings_df
           .merge(products_df[["product_id", "category"]], on="product_id")
           .groupby("category")["rating"].mean()
           .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(10, 5))
colors_rc = [CAT_COLORS.get(c, "#95a5a6") for c in rat_cat.index]
ax.barh(rat_cat.index, rat_cat.values, color=colors_rc, edgecolor="white", linewidth=0.8, height=0.55)
for i, val in enumerate(rat_cat.values):
    ax.text(val + 0.02, i, f"{val:.2f} ★", va="center", fontsize=10, fontweight="bold")
ax.set_title("Average Customer Rating by Category", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Average Rating", fontsize=12)
ax.set_xlim(0, 5.8)
ax.axvline(rat_cat.mean(), color="navy", linewidth=1.5, linestyle="--", alpha=0.6, label=f"Mean ({rat_cat.mean():.2f})")
ax.legend(fontsize=10)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="x", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/05_avg_rating_by_category.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 05_avg_rating_by_category.png")

# ── 06: Purchase Frequency by Age Group ──────────────────────────────────────
age_trans = (trans_df
             .merge(customers_df[["customer_id", "age_group"]], on="customer_id")
             .groupby("age_group")["transaction_id"].count()
             .reindex(["18-24", "25-34", "35-44", "45-54", "55+"]))

fig, ax = plt.subplots(figsize=(9, 5))
colors_a = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
bars = ax.bar(age_trans.index, age_trans.values, color=colors_a,
              edgecolor="white", linewidth=1, width=0.55)
for bar, val in zip(bars, age_trans.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Total Purchases by Customer Age Group", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Age Group", fontsize=12)
ax.set_ylabel("Number of Transactions", fontsize=12)
ax.set_facecolor("#f8f9fa")
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_purchases_by_age.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 06_purchases_by_age.png")

# ── 07: Category Share Pie ────────────────────────────────────────────────────
cat_qty = trans_full.groupby("category")["quantity"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 7))
wedges, texts, autotexts = ax.pie(
    cat_qty.values,
    labels=cat_qty.index,
    autopct="%1.1f%%",
    colors=[CAT_COLORS.get(c, "#95a5a6") for c in cat_qty.index],
    startangle=140,
    pctdistance=0.82,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
for t in autotexts:
    t.set_fontsize(9)
    t.set_fontweight("bold")
ax.set_title("Units Sold Share by Category", fontsize=15, fontweight="bold", pad=16)
plt.tight_layout()
plt.savefig("outputs/07_category_share.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 07_category_share.png")

# ── 08: Revenue Heatmap (City × Category) ────────────────────────────────────
city_cat = (trans_full
            .merge(customers_df[["customer_id", "city"]], on="customer_id")
            .groupby(["city", "category"])["total_price"].sum()
            .unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(city_cat / 1e4, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, annot_kws={"size": 8},
            cbar_kws={"label": "Revenue (₹ × 10K)"})
ax.set_title("Revenue Heatmap: City × Category (₹ ×10K)", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Category", fontsize=11)
ax.set_ylabel("City", fontsize=11)
plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/08_city_category_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ 08_city_category_heatmap.png")

print("\n✅ All 8 visualizations saved to outputs/")
