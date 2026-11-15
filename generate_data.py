"""
Generate a realistic synthetic transaction dataset for the
Sales Performance & Revenue Leakage Analysis.

Business: "Northwind Outfitters" (fictional) – an omnichannel retailer of
outdoor gear selling Online, In-Store and through B2B Sales Reps.

Planted signals the analysis is expected to find:
  1. West region revenue drops ~30% from Jun-2025 (lost B2B rep + competitor).
  2. Sales reps granting discounts above the 15% policy cap.
  3. Expired promo codes still being honoured at checkout.
  4. B2B invoices overdue >90 days (uncollected revenue).
  5. Abandoned online carts (heavier on mobile, at the shipping step).
  6. Returns/refunds concentrated in one product category.

Run:  python generate_data.py   -> writes CSVs to ./data
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
REGIONS = ["North", "South", "East", "West", "Central"]
REGION_WEIGHT = np.array([0.22, 0.18, 0.24, 0.21, 0.15])

# ---------------------------------------------------------------- products
cats = {
    "Tents":       (180, 650, 0.55),
    "Backpacks":   (60, 260, 0.50),
    "Footwear":    (70, 220, 0.52),
    "Apparel":     (25, 140, 0.45),
    "Electronics": (90, 480, 0.68),   # GPS, headlamps – thin margins
    "Accessories": (8, 60, 0.40),
}
rows, pid = [], 1
for cat, (lo, hi, cost_ratio) in cats.items():
    for i in range(10):
        price = round(rng.uniform(lo, hi), 2)
        rows.append(dict(product_id=pid, product_name=f"{cat[:-1] if cat.endswith('s') else cat} Model {chr(65+i)}",
                         category=cat, list_price=price,
                         unit_cost=round(price * rng.normal(cost_ratio, 0.04), 2)))
        pid += 1
products = pd.DataFrame(rows)

# ---------------------------------------------------------------- customers
n_cust = 4000
customers = pd.DataFrame({
    "customer_id": np.arange(1, n_cust + 1),
    "region": rng.choice(REGIONS, n_cust, p=REGION_WEIGHT),
    "segment": rng.choice(["Consumer", "Business"], n_cust, p=[0.82, 0.18]),
    "signup_date": START - pd.to_timedelta(rng.integers(0, 900, n_cust), "D"),
})

# ---------------------------------------------------------------- sales reps
reps = []
rid = 1
for r in REGIONS:
    for k in range(3):
        reps.append(dict(rep_id=rid, rep_name=f"{r[:1]}-Rep{k+1}", region=r,
                         # two reps are habitual over-discounters
                         discount_behaviour="aggressive" if rid in (4, 11) else "normal"))
        rid += 1
reps = pd.DataFrame(reps)
# West rep W-Rep1 (id 10) leaves at end of May 2025
REP_EXIT = {10: pd.Timestamp("2025-05-31")}

# ---------------------------------------------------------------- promo codes
promos = pd.DataFrame([
    dict(promo_code="SPRING10", discount_pct=0.10, valid_from="2024-03-01", valid_to="2024-04-30"),
    dict(promo_code="SUMMER15", discount_pct=0.15, valid_from="2024-06-01", valid_to="2024-07-31"),
    dict(promo_code="BFRIDAY20", discount_pct=0.20, valid_from="2024-11-25", valid_to="2024-12-02"),
    dict(promo_code="WELCOME10", discount_pct=0.10, valid_from="2024-01-01", valid_to="2025-12-31"),
    dict(promo_code="SPRING25", discount_pct=0.10, valid_from="2025-03-01", valid_to="2025-04-30"),
    dict(promo_code="SUMMER25", discount_pct=0.15, valid_from="2025-06-01", valid_to="2025-07-31"),
    dict(promo_code="BFRIDAY25", discount_pct=0.20, valid_from="2025-11-24", valid_to="2025-12-01"),
])
promos[["valid_from", "valid_to"]] = promos[["valid_from", "valid_to"]].apply(pd.to_datetime)

# ---------------------------------------------------------------- orders
days = pd.date_range(START, END, freq="D")
t = np.arange(len(days))
season = 1 + 0.25 * np.sin(2 * np.pi * (days.dayofyear - 80) / 365) \
           + 0.6 * ((days.month == 11) & (days.day >= 20)) + 0.4 * (days.month == 12)
growth = 1 + 0.12 * t / 365
season = np.asarray(season, dtype=float)
daily_n = rng.poisson(62 * season * growth)

order_dates = np.repeat(days.values, daily_n)
n = len(order_dates)
cust = customers.sample(n, replace=True, random_state=1).reset_index(drop=True)
orders = pd.DataFrame({"order_id": np.arange(100001, 100001 + n),
                       "order_date": order_dates,
                       "customer_id": cust.customer_id.values,
                       "region": cust.region.values,
                       "segment": cust.segment.values})

# Planted signal 1: West decline from June 2025 – drop ~30% of West orders
west_drop = (orders.region == "West") & (orders.order_date >= "2025-06-01") & (rng.random(n) < 0.30)
orders = orders[~west_drop].reset_index(drop=True)
n = len(orders)

orders["channel"] = np.where(orders.segment == "Business", "Sales Rep",
                     rng.choice(["Online", "In-Store"], n, p=[0.62, 0.38]))

# assign reps for B2B orders (same region, respecting exits)
orders["rep_id"] = pd.NA
b2b = orders.channel == "Sales Rep"
for region, grp in orders[b2b].groupby("region"):
    ids = reps.loc[reps.region == region, "rep_id"].values
    chosen = rng.choice(ids, len(grp))
    for rep, exit_dt in REP_EXIT.items():
        mask = (chosen == rep) & (grp.order_date.values > np.datetime64(exit_dt))
        chosen[mask] = rng.choice([i for i in ids if i != rep], mask.sum())
    orders.loc[grp.index, "rep_id"] = chosen
orders["rep_id"] = orders.rep_id.astype("Int64")

# ---------------------------------------------------------------- order lines
lines_per = rng.choice([1, 2, 3, 4], n, p=[0.55, 0.27, 0.12, 0.06])
lines_per = np.where(b2b, lines_per + rng.integers(1, 4, n), lines_per)
li = pd.DataFrame({"order_id": np.repeat(orders.order_id.values, lines_per)})
cat_p = np.array([0.12, 0.17, 0.18, 0.25, 0.10, 0.18])
li_cat = rng.choice(list(cats), len(li), p=cat_p)
prod_by_cat = {c: products.product_id[products.category == c].values for c in cats}
li["product_id"] = np.array([prod_by_cat[c][k] for c, k in zip(li_cat, rng.integers(0, 10, len(li)))])
li = li.merge(products[["product_id", "list_price", "unit_cost"]], on="product_id")
li = li.merge(orders[["order_id", "channel"]], on="order_id")
li["quantity"] = np.where(li.channel == "Sales Rep", rng.integers(2, 15, len(li)), rng.choice([1, 1, 1, 2, 3], len(li)))
li = li.drop(columns="channel").sort_values("order_id").reset_index(drop=True)
li.insert(0, "line_id", np.arange(1, len(li) + 1))
li["gross_amount"] = (li.list_price * li.quantity).round(2)

# ---------------------------------------------------------------- discounts
orders["promo_code"] = None
orders["discount_pct"] = 0.0
od = orders.order_date
# promo usage on Online / In-Store orders
consumer = orders.channel != "Sales Rep"
for _, p in promos.iterrows():
    valid = consumer & (od >= p.valid_from) & (od <= p.valid_to)
    use = valid & (rng.random(n) < (0.10 if p.promo_code == "WELCOME10" else 0.35))
    orders.loc[use, ["promo_code", "discount_pct"]] = [p.promo_code, p.discount_pct]
    # Planted signal 3: expired codes still accepted (checkout bug) up to 60 days later
    late = consumer & (od > p.valid_to) & (od <= p.valid_to + pd.Timedelta(days=60)) \
           & orders.promo_code.isna() & (rng.random(n) < 0.06)
    orders.loc[late, ["promo_code", "discount_pct"]] = [p.promo_code, p.discount_pct]

# B2B rep discretionary discounts; policy cap = 15 %
rep_beh = orders.rep_id.map(reps.set_index("rep_id").discount_behaviour)
normal = b2b & (rep_beh == "normal")
aggr = b2b & (rep_beh == "aggressive")
orders.loc[normal, "discount_pct"] = np.clip(rng.normal(0.08, 0.04, normal.sum()), 0, 0.15).round(3)
over = normal & (rng.random(n) < 0.06)   # occasional overrides by anyone
orders.loc[over, "discount_pct"] = rng.uniform(0.16, 0.25, over.sum()).round(3)
orders.loc[aggr, "discount_pct"] = np.clip(rng.normal(0.19, 0.05, aggr.sum()), 0.05, 0.35).round(3)

# ---------------------------------------------------------------- order status & returns
gross = li.groupby("order_id").gross_amount.sum()
orders["gross_amount"] = orders.order_id.map(gross).round(2)
orders["discount_amount"] = (orders.gross_amount * orders.discount_pct).round(2)
orders["net_amount"] = (orders.gross_amount - orders.discount_amount).round(2)
orders["shipping_fee"] = np.where(orders.channel == "Online",
                                  np.where(orders.net_amount >= 100, 0, 7.95), 0)
orders["status"] = rng.choice(["Completed", "Cancelled"], n, p=[0.975, 0.025])
orders["order_date"] = pd.to_datetime(orders.order_date).dt.date

# returns – Planted signal 6: Footwear returns are ~3x the norm (sizing issue)
lc = li.merge(products[["product_id", "category"]], on="product_id") \
       .merge(orders[["order_id", "order_date", "status", "discount_pct"]], on="order_id")
lc = lc[lc.status == "Completed"]
ret_p = np.where(lc.category == "Footwear", 0.16, np.where(lc.category == "Apparel", 0.07, 0.04))
ret = lc[rng.random(len(lc)) < ret_p].copy()
reasons_fw = ["Wrong size", "Wrong size", "Wrong size", "Defective", "Changed mind"]
reasons_ot = ["Defective", "Changed mind", "Not as described", "Wrong size", "Arrived late"]
ret["return_reason"] = [rng.choice(reasons_fw) if c == "Footwear" else rng.choice(reasons_ot) for c in ret.category]
ret["return_qty"] = np.maximum(1, (ret.quantity * rng.uniform(0.3, 1.0, len(ret))).astype(int))
ret["refund_amount"] = (ret.return_qty * ret.list_price * (1 - ret.discount_pct)).round(2)
ret["return_date"] = pd.to_datetime(ret.order_date) + pd.to_timedelta(rng.integers(3, 30, len(ret)), "D")
ret["restockable"] = rng.random(len(ret)) < np.where(ret.return_reason == "Defective", 0.0, 0.7)
returns = ret[["order_id", "line_id", "product_id", "return_date", "return_qty",
               "refund_amount", "return_reason", "restockable"]].reset_index(drop=True)
returns.insert(0, "return_id", np.arange(1, len(returns) + 1))
returns["return_date"] = returns.return_date.dt.date

# ---------------------------------------------------------------- invoices / payments (B2B)
inv = orders[(orders.channel == "Sales Rep") & (orders.status == "Completed")][
    ["order_id", "order_date", "net_amount", "region"]].copy()
inv["invoice_date"] = pd.to_datetime(inv.order_date)
inv["due_date"] = inv.invoice_date + pd.Timedelta(days=30)
# 88 % pay on time-ish, 8 % late, 4 % never (worse in South)
p_never = np.where(inv.region == "South", 0.08, 0.035)
u = rng.random(len(inv))
delay = np.where(u < p_never, np.nan,
         np.where(u < p_never + 0.08, rng.integers(31, 95, len(inv)), rng.integers(5, 30, len(inv))))
inv["paid_date"] = inv.invoice_date + pd.to_timedelta(delay, "D")
inv.loc[inv.paid_date > END, "paid_date"] = pd.NaT
inv["amount_paid"] = np.where(inv.paid_date.notna(), inv.net_amount, 0.0)
# a few partial payments
partial = inv.paid_date.notna() & (rng.random(len(inv)) < 0.03)
inv.loc[partial, "amount_paid"] = (inv.loc[partial, "net_amount"] * rng.uniform(0.5, 0.9, partial.sum())).round(2)
invoices = inv[["order_id", "invoice_date", "due_date", "net_amount", "amount_paid", "paid_date"]] \
    .rename(columns={"net_amount": "invoice_amount"}).reset_index(drop=True)
invoices.insert(0, "invoice_id", np.arange(1, len(invoices) + 1))
for c in ["invoice_date", "due_date", "paid_date"]:
    invoices[c] = pd.to_datetime(invoices[c]).dt.date

# ---------------------------------------------------------------- abandoned carts
n_carts = int((orders.channel == "Online").sum() * 1.9)
cart_days = rng.choice(days.values, n_carts, p=(season * growth) / (season * growth).sum())
cc = customers.sample(n_carts, replace=True, random_state=7)
device = rng.choice(["Desktop", "Mobile", "Tablet"], n_carts, p=[0.38, 0.52, 0.10])
stage_p = {"Desktop": [0.40, 0.25, 0.20, 0.15], "Mobile": [0.30, 0.22, 0.36, 0.12], "Tablet": [0.38, 0.25, 0.24, 0.13]}
stages = ["Cart", "Login", "Shipping", "Payment"]
stage = np.array([rng.choice(stages, p=stage_p[d]) for d in device])
carts = pd.DataFrame({
    "cart_id": np.arange(1, n_carts + 1),
    "created_at": pd.to_datetime(cart_days).date,
    "customer_id": cc.customer_id.values,
    "region": cc.region.values,
    "device": device,
    "items": rng.choice([1, 1, 2, 2, 3, 4], n_carts),
    "abandon_stage": stage,
})
carts["cart_value"] = (carts["items"] * rng.lognormal(np.log(85), 0.6, n_carts)).round(2)
carts["recovered"] = rng.random(n_carts) < 0.08   # recovered by existing reminder email

# ---------------------------------------------------------------- save
for name, df in dict(products=products, customers=customers, sales_reps=reps,
                     promo_codes=promos, orders=orders, order_lines=li,
                     returns=returns, invoices=invoices, abandoned_carts=carts).items():
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:16s} {len(df):>8,} rows")
