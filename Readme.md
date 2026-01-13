
# 🌧️ Rainfall Prediction Classifier  
A complete Machine Learning project for predicting rainfall using weather and climate features. This repository demonstrates the full ML pipeline: data preprocessing, model training, evaluation, prediction API, and Docker-based deployment.

---

## 🚀 Project Overview  .
This project includes:
- Data preprocessing (cleaning, feature engineering, scaling)
- Model training using multiple algorithms
- Model evaluation (accuracy, F1-score, confusion matrix)      
- FastAPI-based prediction service
- Dockerized application for deployment
- Modular and production-ready architecture

---

## 📁 Project Structure
Rainfall_Prediction_Classifier/
│
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ ├── train.py
│ ├── predict.py
│ ├── evaluate.py
│ ├── utils.py
│ └── config.py
│
├── app/
│ ├── api.py
│ ├── cli.py
│ └── init.py
│
├── models/
│ └── rainfall_best_model.joblib
│
├── requirements.txt
├── Dockerfile
└── README.md


---

## 🔧 Tech Stack
- **Python**, **Scikit-Learn**, **NumPy**, **Pandas**
- **FastAPI** + **Uvicorn**
- **Docker**
- **joblib**
- **pytest**

---

## 🧠 Model Training
To train the model:
```bash
python src/train.py

This script:

Loads dataset

Preprocesses and cleans data

Trains ML models

Selects best model

Saves final model in models/ folder

## 📊  Model Evaluting

Run evaluation:
python src/evaluate.py

Outputs include:

Accuracy

Precision/Recall

F1-score

Confusion matrix

## 🌐 Run FastAPI Server Locally

Start API:
uvicorn app.api:app --reload

API documentation:
👉 http://127.0.0.1:8000/docs
 
## 🐳 Run with Docker

Build image:
docker build -t rainfall-api .

Run container:
docker run -p 8000:8000 rainfall-api

API available at:
👉 http://localhost:8000/docs

##📡 Example Prediction Request

POST to /predict:

Request JSON:
{
  "temperature": 25.0,
  "humidity": 72.0,
  "pressure": 1012.3,
  "wind_speed": 2.8
}

Response:
{
  "rainfall_prediction": "Yes"
}
