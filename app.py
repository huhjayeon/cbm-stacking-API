from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 저장된 모델 불러오기
model = joblib.load("stacking_model.pkl")

# 예측 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # JSON 데이터 받기
        input_data = pd.DataFrame([data])  # DataFrame으로 변환
        prediction = model.predict(input_data)  # 예측 수행
        response = {'prediction': prediction.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# 기본 경로
@app.route('/')
def home():
    return "Flask API is running!"

if __name__ == "__main__":
    app.run(debug=True)
