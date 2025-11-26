from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.predict import load_model
from src.config import MODEL_PATH

def evaluate():
    df = load_data()

    X = df.drop("rain_tomorrow", axis=1)
    y = df["rain_tomorrow"]

    model = load_model(MODEL_PATH)
    preds = model.predict(X)

    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate()
