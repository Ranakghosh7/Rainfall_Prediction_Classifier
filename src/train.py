import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data_loader import load_data
from src.preprocess import build_preprocessor
from src.config import MODEL_PATH

def train():
    df = load_data()

    X = df.drop("rain_tomorrow", axis=1)
    y = df["rain_tomorrow"]

    preprocessor = build_preprocessor()

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
