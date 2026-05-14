import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_loader import generate_dataset
from feature_engineering import engineer_features, get_features_and_target, scale_features
from train_models import train_naive_bayes, train_knn, train_decision_tree, evaluate_model, save_models
from visualize import setup, plot_delivery_status, plot_peak_hours, plot_correlation_matrix, plot_geo_distance
from sklearn.model_selection import train_test_split

def main():
    print("="*60)
    print("     FOOD DELIVERY TIME PREDICTION")
    print("="*60)

    print("\n📦 STEP 1: Generating Dataset...")
    df = generate_dataset()

    print("\n🔧 STEP 2: Feature Engineering...")
    df = engineer_features(df)
    X, y, feature_cols = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n🤖 STEP 3: Training Models...")
    nb  = train_naive_bayes(X_train_scaled, y_train)
    knn = train_knn(X_train_scaled, y_train)
    dt  = train_decision_tree(X_train_scaled, y_train)

    print("\n📊 STEP 4: Evaluating Models...")
    print("="*60)
    models = {"naive_bayes": nb, "knn": knn, "decision_tree": dt}
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test_scaled, y_test, name)

    best = max(results, key=results.get)
    print(f"\n🏆 Best Model: {best} with {results[best]*100:.2f}% accuracy")

    print("\n💾 STEP 5: Saving Models...")
    save_models(models)

    print("\n🎨 STEP 6: Generating Visualizations...")
    setup()
    plot_delivery_status(df)
    plot_peak_hours(df)
    plot_correlation_matrix(df)
    plot_geo_distance(df)

    print("\n" + "="*60)
    print("✅ ALL DONE! Check outputs/ folder for models and plots.")
    print("="*60)

if __name__ == "__main__":
    main()