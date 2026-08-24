# =====================================================
# Cairo Metro Transportation Analysis
# Graduation Project
# =====================================================
from pathlib import Path

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

DATA_PATH = Path(".")

hourly = pd.read_csv(DATA_PATH/"agg_avg_passengers_by_time_of_day.csv")

daily_revenue = pd.read_csv(DATA_PATH/"agg_daily_revenue_per_line.csv")

monthly = pd.read_csv(DATA_PATH/"agg_monthly_kpis.csv")

stations = pd.read_csv(DATA_PATH/"agg_passengers_per_station_per_month.csv")

delays = pd.read_csv(DATA_PATH/"agg_delays_per_station.csv")

trip_type = pd.read_csv(DATA_PATH/"agg_revenue_by_trip_type.csv")

weather = pd.read_csv(DATA_PATH/"agg_weather_impact_on_revenue.csv")

od_pairs = pd.read_csv(DATA_PATH/"agg_od_heatmap_top_pairs.csv")

od_monthly = pd.read_csv(DATA_PATH/"agg_od_trips_monthly.csv")

datasets = {
    "Hourly": hourly,
    "Daily Revenue": daily_revenue,
    "Monthly KPI": monthly,
    "Stations": stations,
    "Delays": delays,
    "Trip Type": trip_type,
    "Weather": weather,
    "OD Pairs": od_pairs,
    "OD Monthly": od_monthly
}

for name, df in datasets.items():
    print("="*60)
    print(name)
    print(df.shape)
    print(df.head())

for name, df in datasets.items():

    print("\n")
    print("="*60)
    print(name)

    print(df.info())

    print("\nMissing Values")

    print(df.isnull().sum())

    print("\nDuplicates")

    print(df.duplicated().sum())

for name, df in datasets.items():

    print("="*70)

    print(name)

    print(df.describe(include="all").T)
plt.figure(figsize=(12,6))

bars = plt.bar(
    hourly["hour"],
    hourly["avg_passenger_count"]
)

plt.title("Average Passenger Count by Hour", fontsize=18, weight="bold")
plt.xlabel("Hour")
plt.ylabel("Average Passenger Count")

plt.xticks(hourly["hour"])

for bar in bars:
    y = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        y+0.2,
        f"{y:.1f}",
        ha="center",
        fontsize=9
    )

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("figures/peak_hours.png", dpi=300)

plt.show()
top_station = (
    stations
    .groupby("station_name")["total_passengers"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,7))

top_station.sort_values().plot(kind="barh")

plt.title("Top 10 Busiest Metro Stations", fontsize=18, weight="bold")

plt.xlabel("Passengers")

plt.tight_layout()

plt.savefig("figures/top10_stations.png", dpi=300)

plt.show()


plt.figure(figsize=(12,6))

plt.plot(
    monthly["month"],
    monthly["total_revenue_egp"]/1000000,
    marker="o",
    linewidth=3
)

plt.title("Monthly Revenue")

plt.xlabel("Month")

plt.ylabel("Revenue (Million EGP)")

plt.grid(True)

plt.tight_layout()

plt.savefig("figures/monthly_revenue.png", dpi=300)

plt.show()

plt.figure(figsize=(12,6))

plt.plot(
    monthly["month"],
    monthly["total_passengers"]/1000000,
    marker="o",
    linewidth=3
)

plt.title("Monthly Passenger Volume")

plt.xlabel("Month")

plt.ylabel("Passengers (Million)")

plt.grid()

plt.tight_layout()

plt.savefig("figures/monthly_passengers.png", dpi=300)

plt.show()

corr = monthly.select_dtypes(include="number").corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig("figures/correlation_heatmap.png", dpi=300)

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=weather,
    x="avg_temp_celsius",
    y="avg_passenger_count",
    s=120
)

plt.title("Temperature vs Passenger Count")

plt.xlabel("Temperature (°C)")

plt.ylabel("Average Passenger Count")

plt.tight_layout()

plt.savefig("figures/weather_scatter.png", dpi=300)

plt.show()

top_od = od_pairs.nlargest(10,"total_passengers")

plt.figure(figsize=(14,7))

labels = top_od["origin_station_name"] + " → " + top_od["destination_station_name"]

plt.barh(labels, top_od["total_passengers"])

plt.title("Top 10 Origin-Destination Pairs")

plt.tight_layout()

plt.savefig("figures/top_od_pairs.png", dpi=300)

plt.show()