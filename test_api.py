import requests

# Flask 서버 URL
url = "http://127.0.0.1:5000/predict"

# 예측에 사용할 입력 데이터
input_data = {
    "sku_total_volume": 2.5,
    "sku_count": 5,
    "total_qty": 30
}

# POST 요청 보내기
response = requests.post(url, json=input_data)

# 결과 출력
print("Status Code:", response.status_code)
print("Response:", response.json())
