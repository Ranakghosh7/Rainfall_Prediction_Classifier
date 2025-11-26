# rainfall_classifier_full.py
# Full end-to-end example: generate synthetic data, train pipelines, grid-search RF, evaluate, save model.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib

# -------------------------
# 1) Replace this block with your real dataset load if available:
# df = pd.read_csv("your_rain_data.csv")
# Expected columns example: temperature, humidity, pressure, wind_speed, cloud_cover, month, day, prev_rain_mm, rain_next_day
# -------------------------
np.random.seed(42)
n = 10000
temperature = np.random.normal(20, 6, n)
humidity = np.clip(np.random.normal(75, 15, n),0,100)
pressure = np.random.normal(1013, 8, n)
wind_speed = np.abs(np.random.normal(5, 2.5, n))
cloud_cover = np.clip(np.random.beta(2,2,n)*100,0,100)
month = np.random.randint(1,13,size=n)
day = np.random.randint(1,29,size=n)
prev_rain = np.clip(np.random.exponential(2.0, n),0,100)

logit = (
    0.03*(humidity-50) +
    0.025*(cloud_cover-30) -
    0.02*(pressure-1013) +
    0.1*(prev_rain>1).astype(float) +
    0.02*(10 - np.abs(temperature-18))
)
prob = 1/(1+np.exp(-logit))
rain_next_day = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'pressure': pressure,
    'wind_speed': wind_speed,
    'cloud_cover': cloud_cover,
    'month': month.astype(str),
    'day': day.astype(str),
    'prev_rain_mm': prev_rain,
    'rain_next_day': rain_next_day
})

# Introduce some missingness
for c in ['temperature','humidity','pressure','wind_speed','cloud_cover','prev_rain_mm']:
    df.loc[np.random.rand(n) < 0.02, c] = np.nan

# -------------------------
# 2) Train/test split
# -------------------------
X = df.drop(columns=['rain_next_day'])
y = df['rain_next_day']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## -------------------------
# 3) Preprocessing
# -------------------------
numeric_features = ['temperature','humidity','pressure','wind_speed','cloud_cover','prev_rain_mm']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['month','day']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4) Models and hyperparam search
# -------------------------
rf_pipe = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))])
lr_pipe = Pipeline([('pre', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])

rf_param_grid = {
    'clf__n_estimators': [100, 300],
    'clf__max_depth': [None, 8],
    'clf__min_samples_split': [2, 5]
}

rf_search = GridSearchCV(rf_pipe, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
rf_search.fit(X_train, y_train)   # may take several minutes on large data / wide grid

# Fit logistic baseline
lr_pipe.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

# -------------------------
# 5) Evaluation
# -------------------------
y_prob_rf = best_rf.predict_proba(X_test)[:,1]
y_pred_rf = best_rf.predict(X_test)
y_prob_lr = lr_pipe.predict_proba(X_test)[:,1]
y_pred_lr = lr_pipe.predict(X_test)

print("RF best params:", rf_search.best_params_)
print("RF ROC AUC:", roc_auc_score(y_test, y_prob_rf))
print("LR ROC AUC:", roc_auc_score(y_test, y_prob_lr))
print("\nRandomForest classification report:\n", classification_report(y_test, y_pred_rf))
print("RandomForest confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))

# -------------------------
# 6) Save model & example outputs
# -------------------------
model_path = "rainfall_best_model.joblib"
joblib.dump(best_rf, model_path)
print("Saved model to:", model_path)

# Save a small sample CSV for inspection
df.sample(200, random_state=1).to_csv("sample_rainfall_dataset.csv", index=False)
print("Saved sample CSV: sample_rainfall_dataset.csv")

# Quick predict function usage:
def predict_new(df_new, model_file=model_path):
    model = joblib.load(model_file)
    preds = model.predict(df_new)
    probs = model.predict_proba(df_new)[:,1]
    out = df_new.copy()
    out['predicted_rain'] = preds
    out['predicted_prob'] = probs
    return out

# Example:
ex = predict_new(X_test.iloc[:10])
print(ex)


#for humidity and air speed 

# -------------------------
# 7) Command-line prediction (simple)
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, help="Temperature value")
    parser.add_argument("--hum", type=float, help="Humidity value")
    parser.add_argument("--air", type=float, help="Wind speed / Air speed")

    args = parser.parse_args()

    if args.temp is not None and args.hum is not None and args.air is not None:
        # Create a single-row dataframe for prediction
        df_new = pd.DataFrame([{
            "temperature": args.temp,
            "humidity": args.hum,
            "pressure": 1013,       # default normal value
            "wind_speed": args.air,
            "cloud_cover": 40,      # default
            "month": "5",
            "day": "12",
            "prev_rain_mm": 0       # default
        }])

        result = predict_new(df_new)
        print("\n--- Prediction Result ---")
        print(result)
    else:
        print("\nNo input values provided. To predict:")
        print("python rainfall.py --temp 22 --hum 80 --air 5")

