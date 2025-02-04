from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)

# 특정 도메인 허용
CORS(app, resources={
    r"/*": {"origins": [
        "http://localcms.siliconii.com",
        "https://stgcms.siliconii.com",
        "https://cms.siliconii.com"
    ]}
})

# 전역 스코프에서 model 초기화
model = None

# 저장된 모델 불러오기
model_path = "stacking_model_ver1.pkl"

try:
    model = joblib.load(model_path)  # 모델 로드
    print("모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")

# 예측 API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # JSON 데이터 받기
        input_data = pd.DataFrame([data])  # DataFrame으로 변환
        prediction = model.predict(input_data)  # 예측 수행
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# 기본 경로
@app.route('/')
def home():
    return "Flask API is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
