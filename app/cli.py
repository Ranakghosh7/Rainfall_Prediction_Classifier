import argparse
from src.predict import load_model, predict_df
import pandas as pd




def main():
parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, required=True)
parser.add_argument('--humidity', type=float, required=True)
parser.add_argument('--pressure', type=float, required=True)
parser.add_argument('--wind_speed', type=float, required=True)
parser.add_argument('--cloud_cover', type=float, required=True)
parser.add_argument('--month', type=int, required=True)
parser.add_argument('--day', type=int, required=True)
parser.add_argument('--prev_rain_mm', type=float, default=0.0)
parser.add_argument('--model', type=str, default='models/rainfall_best_model.joblib')
args = parser.parse_args()


model = load_model(args.model)
df = pd.DataFrame([{
'temperature': args.temperature,
'humidity': args.humidity,
'pressure': args.pressure,
'wind_speed': args.wind_speed,
'cloud_cover': args.cloud_cover,
'month': str(args.month),
'day': str(args.day),
'prev_rain_mm': args.prev_rain_mm
}])


out = predict_df(model, df)
print(out.to_string(index=False))


if __name__ == '__main__':
main()