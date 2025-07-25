import pandas as pd

# Dummy structured data
structured_data = pd.DataFrame([
    {"product": "Product A", "region": "North", "revenue": 60000, "online_pct": 0.25, "profit_margin": 0.20, "growth_qoq": 0.14, "avg_order_value": 120},
    {"product": "Product A", "region": "South", "revenue": 55000, "online_pct": 0.30, "profit_margin": 0.18, "growth_qoq": 0.12, "avg_order_value": 116},
    {"product": "Product B", "region": "East", "revenue": 40000, "online_pct": 0.28, "profit_margin": 0.17, "growth_qoq": 0.10, "avg_order_value": 112},
    {"product": "Product B", "region": "West", "revenue": 45000, "online_pct": 0.27, "profit_margin": 0.19, "growth_qoq": 0.11, "avg_order_value": 115},
    {"product": "Product C", "region": "North", "revenue": 47855, "online_pct": 0.33, "profit_margin": 0.22, "growth_qoq": 0.13, "avg_order_value": 125},
    {"product": "Product C", "region": "East", "revenue": 45000, "online_pct": 0.35, "profit_margin": 0.21, "growth_qoq": 0.15, "avg_order_value": 127},
])