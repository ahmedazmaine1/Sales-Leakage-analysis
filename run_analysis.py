"""
Sales Performance & Revenue Leakage Analysis — pipeline.

  1. Load raw CSVs into a SQLite warehouse        (output/sales.db)
  2. Run every query in sql/analysis_queries.sql  (output/query_results/*.csv)
  3. Pandas layer: leakage sizing, recovery model, headline KPIs
  4. Excel report with live formulas + charts     (output/Revenue_Leakage_Report.xlsx)
  5. Interactive HTML dashboard                   (dashboard/index.html)

Run:  python generate_data.py && python run_analysis.py
"""
from pathlib import Path
import json, re, sqlite3
import pandas as pd

ROOT = Path(__file__).parent
DATA, OUT, SQL = ROOT / "data", ROOT / "output", ROOT / "sql" / "analysis_queries.sql"
(OUT / "query_results").mkdir(parents=True, exist_ok=True)

# Recovery assumptions: share of each leak the business can realistically win back.
RECOVERY = {
    "Excess rep discounts (>15% cap)": (0.80, "Enforce manager approval above 15% in CPQ"),
    "Expired promo codes honoured":    (1.00, "Validate promo valid_from/valid_to at checkout"),
    "Uncollected receivables (>90d)":  (0.40, "Collections agency + credit holds on repeat offenders"),
    "High-intent abandoned carts":     (0.15, "Cart-reminder flow + simplify mobile shipping step"),
    "Excess returns (above baseline)": (0.50, "Size guide / fit tool on Footwear PDPs"),
}

# ------------------------------------------------------------------ 1. load
db = OUT / "sales.db"
db.unlink(missing_ok=True)
con = sqlite3.connect(db)
for csv in sorted(DATA.glob("*.csv")):
    pd.read_csv(csv).to_sql(csv.stem, con, index=False)
con.executescript("""
CREATE INDEX ix_orders_date   ON orders(order_date);
CREATE INDEX ix_orders_region ON orders(region);
CREATE INDEX ix_lines_order   ON order_lines(order_id);
CREATE INDEX ix_inv_order     ON invoices(order_id);
""")

# ------------------------------------------------------------------ 2. SQL
blocks = re.split(r"^-- name:\s*(\w+)\s*$", SQL.read_text(), flags=re.M)
Q = {}
for name, body in zip(blocks[1::2], blocks[2::2]):
    Q[name] = pd.read_sql_query(body, con)
    Q[name].to_csv(OUT / "query_results" / f"{name}.csv", index=False)
print(f"Ran {len(Q)} SQL queries -> output/query_results/")

# ------------------------------------------------------------------ 3. pandas layer
monthly = Q["monthly_revenue"]
monthly["year"] = monthly.month.str[:4]
fy = monthly.groupby("year")[["gross_revenue", "discounts", "net_revenue", "refunds", "realized_revenue", "orders"]].sum()
net_25, net_24 = fy.loc["2025", "net_revenue"], fy.loc["2024", "net_revenue"]

returns = Q["leak_returns"]
baseline = returns.return_rate_pct.median()
excess_ret = returns[returns.return_rate_pct > baseline]
excess_returns = float(((excess_ret.return_rate_pct - baseline) / excess_ret.return_rate_pct * excess_ret.refunds).sum()
                       + excess_ret.written_off_cost.sum())

carts = Q["leak_abandoned_carts"]
leak = pd.DataFrame({
    "leak_type": list(RECOVERY),
    "amount_24m": [
        Q["leak_excess_discount"].leakage.sum(),
        Q["leak_expired_promo"].leakage.sum(),
        Q["leak_uncollected"].over_90_leakage.sum(),
        carts[carts.high_intent == 1].lost_value.sum(),
        excess_returns,
    ],
})
leak["recovery_rate"] = leak.leak_type.map(lambda k: RECOVERY[k][0])
leak["recoverable"] = leak.amount_24m * leak.recovery_rate
leak["pct_of_net_revenue"] = 100 * leak.amount_24m / monthly.net_revenue.sum()
leak["action"] = leak.leak_type.map(lambda k: RECOVERY[k][1])
leak = leak.sort_values("recoverable", ascending=False).reset_index(drop=True)
leak.to_csv(OUT / "leakage_summary.csv", index=False)

drop = Q["regional_drop_detection"]
flagged = drop[drop.flag == "DROP"]
rc = Q["region_channel_drop"]
reps = Q["rep_performance"]

kpi = {
    "net_rev_2025": net_25,
    "net_rev_2024": net_24,
    "yoy_pct": 100 * (net_25 / net_24 - 1),
    "orders_2025": int(fy.loc["2025", "orders"]),
    "aov_2025": net_25 / fy.loc["2025", "orders"],
    "leak_total": leak.amount_24m.sum(),
    "leak_recoverable": leak.recoverable.sum(),
    "leak_pct": 100 * leak.amount_24m.sum() / monthly.net_revenue.sum(),
    "return_baseline": baseline,
}

findings = []
for _, r in flagged.iterrows():
    worst = rc[rc.region == r.region].sort_values("yoy_pct").iloc[0]
    findings.append(f"{r.region} region H2-2025 revenue is {r.yoy_pct:+.1f}% YoY vs "
                    f"{drop[drop.region != r.region].yoy_pct.mean():+.1f}% for the other regions — about ${-r.gap_vs_peer_growth:,.0f} below peer-growth trend. "
                    f"Worst channel: {worst.channel} ({worst.yoy_pct:+.1f}%).")
gone = reps[pd.to_datetime(reps.last_order_date) < "2025-07-01"]
for _, r in gone.iterrows():
    findings.append(f"Rep {r.rep_name} ({r.region}) has had no orders since {r.last_order_date} — accounts were not fully reassigned.")
top2 = reps.head(2)
findings.append(f"Reps {' & '.join(top2.rep_name)} account for {100*top2.excess_discount.sum()/reps.excess_discount.sum():.0f}% "
                f"of above-cap discounting (${top2.excess_discount.sum():,.0f}); "
                f"{top2.pct_orders_over_cap.mean():.0f}% of their orders exceed the 15% cap.")
ep = Q["leak_expired_promo"]
findings.append(f"{ep.invalid_redemptions.sum():,} orders redeemed expired promo codes (${ep.leakage.sum():,.0f}) — checkout does not validate expiry.")
ar = Q["leak_uncollected"]
findings.append(f"${ar.over_90_leakage.sum():,.0f} of B2B invoices are >90 days past due; "
                f"{ar.iloc[0].region} alone holds ${ar.iloc[0].over_90_leakage:,.0f}.")
stage = carts.groupby("abandon_stage").lost_value.sum().sort_values(ascending=False)
mob_ship = carts[(carts.device == "Mobile") & (carts.abandon_stage == "Shipping")].lost_value.sum()
hi = carts[carts.high_intent == 1].lost_value.sum()
findings.append(f"${hi:,.0f} in high-intent carts abandoned at Shipping/Payment; Mobile at the Shipping step alone loses ${mob_ship:,.0f} "
                f"({100*mob_ship/hi:.0f}%) — the mobile shipping form is the single biggest checkout leak.")
fw = returns.iloc[0]
findings.append(f"{fw.category} return rate is {fw.return_rate_pct:.1f}% vs {baseline:.1f}% baseline; "
                f"{fw.pct_wrong_size:.0f}% of those returns cite 'Wrong size'.")
(OUT / "key_findings.txt").write_text("\n".join(f"{i}. {f}" for i, f in enumerate(findings, 1)))

print(f"\nFY2025 net revenue ${net_25:,.0f} ({kpi['yoy_pct']:+.1f}% YoY)")
print(f"Total leakage (24m) ${kpi['leak_total']:,.0f} = {kpi['leak_pct']:.1f}% of net revenue; "
      f"recoverable ≈ ${kpi['leak_recoverable']:,.0f}\n")
print(leak[["leak_type", "amount_24m", "recovery_rate", "recoverable"]].to_string(index=False))
print("\nKEY FINDINGS"); print((OUT / "key_findings.txt").read_text())

# ------------------------------------------------------------------ 4. Excel report
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HDR = PatternFill("solid", fgColor="1C5CAB")
wb = Workbook()

def sheet(title, df, money=(), pct=(), widths=None):
    ws = wb.create_sheet(title)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row])
    for c in ws[1]:
        c.font, c.fill, c.alignment = Font(bold=True, color="FFFFFF"), HDR, Alignment(wrap_text=True, vertical="center")
    for j, col in enumerate(df.columns, 1):
        L = get_column_letter(j)
        ws.column_dimensions[L].width = max(12, min(48, len(col) + 4))
        fmt = '"$"#,##0' if col in money else ('0.0"%"' if col in pct else None)
        if fmt:
            for cell in ws[L][1:]:
                cell.number_format = fmt
    ws.freeze_panes = "A2"
    return ws

# Executive summary
ws = wb.active; ws.title = "Executive Summary"
ws["A1"] = "Sales Performance & Revenue Leakage Analysis"; ws["A1"].font = Font(size=16, bold=True)
ws["A2"] = "Northwind Outfitters (synthetic data) · Jan 2024 – Dec 2025 · as of 31-Dec-2025"
rows = [("FY2025 net revenue", kpi["net_rev_2025"], '"$"#,##0'),
        ("FY2024 net revenue", kpi["net_rev_2024"], '"$"#,##0'),
        ("YoY growth", "=B4/B5-1", "0.0%"),
        ("FY2025 orders", kpi["orders_2025"], "#,##0"),
        ("FY2025 avg order value", kpi["aov_2025"], '"$"#,##0.00'),
        ("Revenue leakage (24 months)", "='Leakage Summary'!B7", '"$"#,##0'),
        ("Recoverable leakage", "='Leakage Summary'!D7", '"$"#,##0')]
for i, (k, v, f) in enumerate(rows, 4):
    ws[f"A{i}"], ws[f"B{i}"] = k, v
    ws[f"B{i}"].number_format = f; ws[f"A{i}"].font = Font(bold=True)
ws["A13"] = "Key findings"; ws["A13"].font = Font(size=13, bold=True)
for i, f in enumerate(findings, 14):
    ws[f"A{i}"] = f"{i-13}. {f}"
ws.column_dimensions["A"].width = 34; ws.column_dimensions["B"].width = 18

# Leakage summary with live formulas
ws = wb.create_sheet("Leakage Summary")
ws.append(["Leak type", "Amount (24m)", "Recovery rate", "Recoverable", "% of net revenue", "Recommended action"])
for i, r in enumerate(leak.itertuples(), 2):
    ws.append([r.leak_type, round(r.amount_24m, 2), r.recovery_rate, f"=B{i}*C{i}",
               f"=B{i}/{monthly.net_revenue.sum():.2f}", r.action])
ws.append(["TOTAL", "=SUM(B2:B6)", "=D7/B7", "=SUM(D2:D6)", "=SUM(E2:E6)", ""])
for c in ws[1]:
    c.font, c.fill = Font(bold=True, color="FFFFFF"), HDR
for c in ws[7]:
    c.font = Font(bold=True)
for i in range(2, 8):
    ws[f"B{i}"].number_format = ws[f"D{i}"].number_format = '"$"#,##0'
    ws[f"C{i}"].number_format = ws[f"E{i}"].number_format = "0.0%"
for L, w in zip("ABCDEF", (36, 16, 14, 16, 16, 58)):
    ws.column_dimensions[L].width = w
ws["A9"] = "Edit the Recovery rate column to stress-test the recoverable estimate."
ch = BarChart(); ch.type = "bar"; ch.title = "Recoverable leakage by type"; ch.legend = None
ch.add_data(Reference(ws, min_col=4, min_row=1, max_row=6), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=6))
ch.height, ch.width = 7.5, 18
ws.add_chart(ch, "A11")

money_cols = {"gross_revenue", "discounts", "net_revenue", "refunds", "realized_revenue", "avg_order_value",
              "h2_2024", "h1_2025", "h2_2025", "gap_vs_peer_growth", "excess_discount", "leakage",
              "not_yet_due", "d1_30", "d31_90", "over_90_leakage", "abandoned_value", "recovered_value",
              "lost_value", "written_off_cost", "net_sales", "cogs"}
pct_cols = {"mom_growth_pct", "yoy_growth_pct", "yoy_pct", "seq_pct", "avg_discount_pct",
            "pct_orders_over_cap", "return_rate_pct", "pct_wrong_size", "margin_pct"}

ws = sheet("Monthly Revenue", monthly.drop(columns="year"), money_cols, pct_cols)
lc = LineChart(); lc.title = "Monthly net vs realized revenue"; lc.height, lc.width = 8, 22
lc.add_data(Reference(ws, min_col=5, min_row=1, max_row=len(monthly) + 1), titles_from_data=True)
lc.add_data(Reference(ws, min_col=7, min_row=1, max_row=len(monthly) + 1), titles_from_data=True)
lc.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(monthly) + 1))
ws.add_chart(lc, "M2")

pivot = Q["regional_monthly"].pivot(index="month", columns="region", values="net_revenue").reset_index()
ws = sheet("Regional Trend", pivot, set(pivot.columns) - {"month"})
lc = LineChart(); lc.title = "Net revenue by region"; lc.height, lc.width = 8, 22
lc.add_data(Reference(ws, min_col=2, max_col=6, min_row=1, max_row=len(pivot) + 1), titles_from_data=True)
lc.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(pivot) + 1))
ws.add_chart(lc, "H2")
sheet("Regional Drop", drop, money_cols, pct_cols)
sheet("Region x Channel", rc, money_cols, pct_cols)
sheet("Rep Scorecard", reps, money_cols, pct_cols)
sheet("Expired Promos", ep, money_cols, pct_cols)
sheet("AR Aging", ar, money_cols, pct_cols)
sheet("Abandoned Carts", carts, money_cols, pct_cols)
sheet("Returns", returns, money_cols, pct_cols)
sheet("Category Margin", Q["category_margin"], money_cols, pct_cols)
xlsx = OUT / "Revenue_Leakage_Report.xlsx"
wb.save(xlsx)
print(f"\nExcel report -> {xlsx.relative_to(ROOT)}")

# ------------------------------------------------------------------ 5. dashboard data
dash = {
    "kpi": kpi,
    "findings": findings,
    "monthly": monthly[["month", "net_revenue", "realized_revenue", "discounts", "refunds", "orders"]].to_dict("list"),
    "regional": {r: g.net_revenue.round(0).tolist() for r, g in Q["regional_monthly"].groupby("region")},
    "regional_months": sorted(Q["regional_monthly"].month.unique()),
    "drop": drop.to_dict("records"),
    "leak": leak.round(2).to_dict("records"),
    "reps": reps[["rep_name", "region", "net_revenue", "avg_discount_pct", "pct_orders_over_cap", "excess_discount"]].to_dict("records"),
    "ar": ar.to_dict("records"),
    "carts": carts.groupby(["abandon_stage", "device"]).lost_value.sum().unstack().reindex(["Cart", "Login", "Shipping", "Payment"]).round(0).to_dict(),
    "returns": returns[["category", "return_rate_pct", "refunds", "pct_wrong_size"]].to_dict("records"),
}
tpl = (ROOT / "dashboard" / "template.html").read_text()
(ROOT / "dashboard" / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(dash, default=float)))
print("Dashboard    -> dashboard/index.html")
con.close()
