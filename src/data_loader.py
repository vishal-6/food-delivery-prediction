import pandas as pd
import numpy as np
import os

def generate_dataset(n_samples=10000, save_path="data/food_delivery.csv"):
    np.random.seed(42)

    data = {
        "order_id":             range(1, n_samples + 1),
        "restaurant_lat":       np.random.uniform(12.85, 13.15, n_samples),
        "restaurant_lon":       np.random.uniform(77.45, 77.75, n_samples),
        "customer_lat":         np.random.uniform(12.85, 13.15, n_samples),
        "customer_lon":         np.random.uniform(77.45, 77.75, n_samples),
        "order_hour":           np.random.randint(0, 24, n_samples),
        "order_day":            np.random.randint(0, 7, n_samples),
        "order_month":          np.random.randint(1, 13, n_samples),
        "weather":              np.random.choice(["Clear", "Rainy", "Foggy", "Windy"], n_samples),
        "traffic":              np.random.choice(["Low", "Medium", "High"], n_samples),
        "vehicle_type":         np.random.choice(["Bike", "Scooter", "Car"], n_samples),
        "num_items":            np.random.randint(1, 10, n_samples),
        "restaurant_rating":    np.random.uniform(1, 5, n_samples).round(1),
        "delivery_person_age":  np.random.randint(18, 50, n_samples),
        "delivery_person_exp":  np.random.randint(0, 10, n_samples),
    }

    df = pd.DataFrame(data)

    delay_score = (
        (df["traffic"] == "High").astype(int) * 2 +
        (df["weather"] == "Rainy").astype(int) * 1.5 +
        (df["weather"] == "Foggy").astype(int) * 1 +
        (df["order_hour"].between(12, 14) | df["order_hour"].between(19, 21)).astype(int) * 1.5 +
        (df["num_items"] > 5).astype(int) * 1 +
        np.random.normal(0, 0.5, n_samples)
    )

    df["delivery_status"] = pd.cut(
        delay_score,
        bins=[-np.inf, 1.5, 3.5, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved to {save_path} — {n_samples} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print("\nDelivery Status Distribution:")
    print(df["delivery_status"].value_counts())