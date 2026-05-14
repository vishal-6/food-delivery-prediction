import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def engineer_features(df):
    df["geo_distance_km"] = haversine_distance(
        df["restaurant_lat"], df["restaurant_lon"],
        df["customer_lat"],   df["customer_lon"]
    )

    df["is_peak_hour"] = (
        df["order_hour"].between(12, 14) |
        df["order_hour"].between(19, 21)
    ).astype(int)

    df["is_weekend"] = (df["order_day"] >= 5).astype(int)

    df["is_rainy_or_foggy"] = (
        df["weather"].isin(["Rainy", "Foggy"])
    ).astype(int)

    le = LabelEncoder()
    df["weather_enc"]  = le.fit_transform(df["weather"])
    df["traffic_enc"]  = le.fit_transform(df["traffic"])
    df["vehicle_enc"]  = le.fit_transform(df["vehicle_type"])

    print("✅ Features engineered: geo_distance_km, is_peak_hour, is_weekend, is_rainy_or_foggy")
    return df

def get_features_and_target(df):
    feature_cols = [
        "geo_distance_km", "order_hour", "order_day", "order_month",
        "weather_enc", "traffic_enc", "vehicle_enc",
        "num_items", "restaurant_rating", "delivery_person_age",
        "delivery_person_exp", "is_peak_hour", "is_weekend", "is_rainy_or_foggy"
    ]
    X = df[feature_cols]
    y = df["delivery_status"]
    return X, y, feature_cols

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print(f"✅ Features scaled — {X_train_scaled.shape[1]} features")
    return X_train_scaled, X_test_scaled, scaler