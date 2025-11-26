import joblib
import pandas as pd

def load_model(path):
    return joblib.load(path)

def predict_df(model, df: pd.DataFrame):
    preds = model.predict(df)
    df["prediction"] = preds
    return df
