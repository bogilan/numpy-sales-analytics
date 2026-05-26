import numpy as np
import matplotlib.pyplot as plt

# =========================
# SALES ANALYTICS PROJECT
# =========================

# Reproducible random data
np.random.seed(42)

# =========================
# TASK SETUP
# =========================

# Generate 365 days and 5 products sales
days = 365
products = 5

product_names = [
    "Laptop",
    "Phone",
    "Tablet",
    "Monitor",
    "Keyboard"
]

# =========================
# GENERATE SALES DATA
# =========================
# Base random sales data
sales = np.random.randint(100, 500, size=(days, products))

# Add seasonality trend
# Create 365 evenly spaced numbers between 0 and 12 for smooth value increasing
even_space_range = np.linspace(0, 12, days)

# Produce wave patern values for visualization and scale it by 50
wave_patern = np.sin(even_space_range) * 50

# Reshape it to 2D array
seasonal_trend = (wave_patern).reshape(-1, 1)

# Add trend
sales = sales + seasonal_trend

# Convert to integers
sales = sales.astype(int)


# =========================
# KPI CALCULATIONS
# =========================

# Total yearly sales
total_sales = np.sum(sales)

# Total sales per day
daily_totals = np.sum(sales, axis=1)

# axis=0 -> calculate vertically (per product)
product_totals = np.sum(sales, axis=0)

# Average daily sales
average_daily_sales = np.mean(daily_totals)

# Maximum sales value for each product
max_sales_per_product = np.max(sales, axis=0)

# Average sales per product
average_sales_per_product = np.mean(sales, axis=0)

# =========================
# BEST/WORST PERFORMANCE
# =========================

best_day = np.argmax(daily_totals)
best_day_sale = np.max(daily_totals)
worst_day = np.argmin(daily_totals)
worst_day_sale = np.min(daily_totals)

best_product = np.argmax(product_totals)
worst_product = np.argmin(product_totals)

# =========================
# MOVING AVERAGE
# =========================

moving_average_7d = np.convolve(
    daily_totals,
    np.ones(7) / 7,
    mode='valid'
)

# =========================
# ANOMALY DETECTION
# =========================

daily_std = np.std(daily_totals)

threshold = average_daily_sales + 2 * daily_std

anomaly_days = np.where(daily_totals > threshold)[0]

# =========================
# PRODUCT RANKING
# =========================

# Ranks products by total sale
# [::-1] -> in descending order
ranking = np.argsort(product_totals)[::-1]


# =========================
# BUSINESS INSIGHTS
# =========================

# Headline
print("\n========== SALES REPORT ==========\n")

print(f"Total yearly sales: {total_sales:,}")
print(f"Average daily sales: {average_daily_sales:.2f}")

print(
    f"\nBest sales day: Day {best_day}"
    f", with sale of: {best_day_sale:,} units"
)

print(
    f"Worst sales day: Day {worst_day}"
    f",  with sale of: {worst_day_sale:,} units"
)

print(
    f"\nBest selling product: "
    f"{product_names[best_product]}"
)

print(
    f"Worst selling product: "
    f"{product_names[worst_product]}"
)

print(
    f"\nNumber of anomaly days detected: "
    f"{len(anomaly_days)}"
)

above_average_days = np.sum(
    daily_totals > average_daily_sales
)

print(
    f"Days above average sales: "
    f"{above_average_days}"
)


# =========================
# PRODUCT PERFORMANCE
# =========================

print("\n========== PRODUCT PERFORMANCE ==========\n")

for i in range(products):
    print(
        f"{product_names[i]}:"
        f"\nTotal Sales: {product_totals[i]:,}"
        f"\n Average Daily Sales: {average_sales_per_product[i]:.2f}"
        f"\n Maximum Daily Sale: {max_sales_per_product[i]}"
        f"\n"
    )

# =========================
# PRODUCT RANKING
# =========================

print("\n========== PRODUCT RANKING ==========\n")

# enumerate() gives rank number to product indexes (idx) within provided ranking
for rank, idx in enumerate(ranking, start=1):
    print(
        f"{rank}. {product_names[idx]} - {product_totals[idx]:,} units sold"
    )

# =========================
# VISUALIZATIONS
# =========================

# Daily sales trend
plt.figure(figsize=(14, 6))

plt.plot(
    daily_totals,
    label="Daily Sales"
)

plt.plot(
    range(6, days),
    moving_average_7d,
    linewidth=3,
    label="7-Day Moving Average"
)

# Highlight anomaly days
plt.scatter(
    anomaly_days,
    daily_totals[anomaly_days],
    label="Anomaly Days"
)

plt.title("Daily Sales Trend")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.legend()

plt.savefig("images/sales_trend.png")
plt.close()

# =========================
# PRODUCT COMPARISON CHART
# =========================

plt.figure(figsize=(10, 6))

plt.bar(
    product_names,
    product_totals
)

plt.title("Total Sales per Product")
plt.xlabel("Products")
plt.ylabel("Total Sales")

plt.savefig("images/product_sales.png")
plt.close()