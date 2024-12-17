import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression



# 1. 환경 변수 로드 및 확인
def load_env_variables():
    load_dotenv()
    host = os.getenv("REDSHIFT_HOST")
    port = os.getenv("REDSHIFT_PORT")
    dbname = os.getenv("REDSHIFT_DBNAME")
    user = os.getenv("REDSHIFT_USER")
    password = os.getenv("REDSHIFT_PASSWORD")

    if not all([host, port, dbname, user, password]):
        raise ValueError("환경 변수 값이 올바르지 않습니다. .env 파일을 확인하세요.")

    return host, port, dbname, user, password




# 2. Redshift 연결 및 데이터 로드

def load_data_from_redshift(host, port, dbname, user, password, query_cbm, query_prod):
    try:
        conn = psycopg2.connect(
            host=host,
            port=int(port),
            dbname=dbname,
            user=user,
            password=password
        )
        print("Redshift 연결 성공!")

        pckg_cbm_df = pd.read_sql(query_cbm, conn)
        pckg_prod_df = pd.read_sql(query_prod, conn)
        print("테이블 가져오기 성공!")

        return pckg_cbm_df, pckg_prod_df

    except Exception as e:
        print("데이터 로드 중 오류 발생:", e)
        raise

    finally:
        if conn:
            conn.close()
            print("Redshift 연결 종료")


# 3. 데이터 전처리 함수
def preprocess_data(pckg_cbm_df, pckg_prod_df):
    # SKU 종류와 수량 계산
    sku_count = pckg_prod_df.groupby('pckg_no')['prod_cd'].nunique().reset_index()
    sku_count.rename(columns={'prod_cd': 'sku_count'}, inplace=True)

    total_qty = pckg_prod_df.groupby('pckg_no')['qty'].sum().reset_index()
    total_qty.rename(columns={'qty': 'total_qty'}, inplace=True)

    # SKU 총 부피 계산
    sku_total_volume = pckg_prod_df.groupby('pckg_no')['total_prod_volume'].sum().reset_index()
    sku_total_volume.rename(columns={'total_prod_volume': 'sku_total_volume'}, inplace=True)

    # 패킹 데이터와 병합
    pallet_total_cbm = pckg_cbm_df[['pckg_no', 'total_pckg_cbm']].drop_duplicates()
    merged_data = pd.merge(sku_total_volume, pallet_total_cbm, on='pckg_no')
    merged_data = pd.merge(merged_data, sku_count, on='pckg_no')
    merged_data = pd.merge(merged_data, total_qty, on='pckg_no')

    # 팔레트 높이 추가
    merged_data['pallet_height'] = merged_data['total_pckg_cbm'] / (1.1 * 1.1)  # 가로와 세로는 1.1m 고정

    # 패킹넘버별 팔레트 개수 계산
    pallet_counts = pckg_cbm_df.groupby('pckg_no').size().reset_index(name='pallet_count')

    # 팔레트 개수와 최대 가능한 부피 계산
    pallet_counts['max_possible_cbm'] = pallet_counts['pallet_count'] * (1.1 * 1.1 * 2.4)

    # 패킹 데이터와 병합
    merged_with_counts = pd.merge(merged_data, pallet_counts, on='pckg_no')

    # 이상치 제거
    filtered_data = merged_with_counts[
        merged_with_counts['total_pckg_cbm'] <= merged_with_counts['max_possible_cbm']]

    return filtered_data


# 4. 모델 학습 함수
def train_stacking_model(data):
    X = data[['sku_total_volume', 'sku_count', 'total_qty']]
    y = data['total_pckg_cbm']

    # 하한값 설정
    lower_bound = 1.21
    valid_indices = y[y > lower_bound].index
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    # 학습 및 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 모델 설정
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    lgbm_model = LGBMRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)

    meta_model = LinearRegression()

    stacking_model = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgbm', lgbm_model)
        ],
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    # 모델 학습
    stacking_model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(stacking_model, 'stacking_model.pkl')
    print("모델이 성공적으로 저장되었습니다.")


# 5. 메인 함수
def main():
    # 환경 변수 로드
    host, port, dbname, user, password = load_env_variables()

    # SQL 쿼리
    query_cbm = 'SELECT * FROM "dev"."cms"."cms_mysql_tb_pckg_cbm";'
    query_prod = 'SELECT * FROM "dev"."cms"."cms_mysql_tb_pckg_prod";'

    # 데이터 로드
    pckg_cbm_df, pckg_prod_df = load_data_from_redshift(host, port, dbname, user, password, query_cbm, query_prod)

    # 데이터 전처리
    processed_data = preprocess_data(pckg_cbm_df, pckg_prod_df)
    print("데이터 전처리 완료")

    # 모델 학습
    train_stacking_model(processed_data)


if __name__ == "__main__":
    main()
