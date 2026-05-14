import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import generate_dataset
from feature_engineering import engineer_features

def setup():
    os.makedirs("outputs", exist_ok=True)
    sns.set_theme(style="whitegrid")

def plot_delivery_status(df):
    print("📊 Plotting delivery status distribution...")
    plt.figure(figsize=(8, 5))
    labels = {0: "On Time", 1: "Delayed", 2: "Very Late"}
    df["status_label"] = df["delivery_status"].map(labels)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    sns.countplot(data=df, x="status_label", order=["On Time", "Delayed", "Very Late"],
                  hue="status_label", palette=colors, legend=False)
    plt.title("Delivery Status Distribution", fontsize=16)
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/delivery_status.png", dpi=150)
    plt.close()
    print("✅ Saved → outputs/delivery_status.png")

def plot_peak_hours(df):
    print("📊 Plotting peak hour delays...")
    plt.figure(figsize=(12, 5))
    hour_delay = df.groupby("order_hour")["delivery_status"].mean()
    hour_delay.plot(kind="bar", color="steelblue")
    plt.title("Average Delay Score by Order Hour", fontsize=16)
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Delivery Status")
    plt.tight_layout()
    plt.savefig("outputs/peak_hours.png", dpi=150)
    plt.close()
    print("✅ Saved → outputs/peak_hours.png")

def plot_correlation_matrix(df):
    print("📊 Plotting correlation matrix...")
    num_cols = [
        "geo_distance_km", "order_hour", "num_items",
        "restaurant_rating", "delivery_person_age",
        "delivery_person_exp", "is_peak_hour",
        "is_weekend", "delivery_status"
    ]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/correlation_matrix.png", dpi=150)
    plt.close()
    print("✅ Saved → outputs/correlation_matrix.png")

def plot_geo_distance(df):
    print("📊 Plotting geo distance vs delivery status...")
    plt.figure(figsize=(8, 5))
    labels = {0: "On Time", 1: "Delayed", 2: "Very Late"}
    df["status_label"] = df["delivery_status"].map(labels)
    sns.boxplot(data=df, x="status_label", y="geo_distance_km",
                order=["On Time", "Delayed", "Very Late"],
                hue="status_label",
                palette=["#2ecc71", "#f39c12", "#e74c3c"],
                legend=False)
    plt.title("Geo Distance vs Delivery Status", fontsize=16)
    plt.xlabel("Delivery Status")
    plt.ylabel("Distance (km)")
    plt.tight_layout()
    plt.savefig("outputs/geo_distance.png", dpi=150)
    plt.close()
    print("✅ Saved → outputs/geo_distance.png")

if __name__ == "__main__":
    setup()
    df = generate_dataset()
    df = engineer_features(df)

    plot_delivery_status(df)
    plot_peak_hours(df)
    plot_correlation_matrix(df)
    plot_geo_distance(df)

    print("\n✅ All plots saved to outputs/ folder!")