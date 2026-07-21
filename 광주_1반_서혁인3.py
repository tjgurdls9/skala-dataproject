"""
파일명: 광주_1반_서혁인3.py
프로그램 설명: Pandas, Polars Lazy API, DuckDB SQL 성능 비교 및 EDA 실습
프로그램 설명 비고: 데이터 파일이 경로에 없을 때의 오류/예외 처리를 위해(혹은 데이터 없이 코드 테스팅을 위해) 임의의 10만 건 테스트 데이터를 생성하여 작업을 실행하도록 함.
                실습을 위한 데이터인 sales_100k.csv 외에 다른 데이터에 대한 작업을 진행할 경우, 파이썬을 Run할 때에 터미널 창에 파일명을 같이 입력하는 방법으로
                프로그램의 유연성 및 실무 적용도를 제고함.
                위 과정에서, 생성된 테스트 데이터 파일은 폴더에 저장하도록 하여 테스팅 결과 분석에 활용할 수 있게 함.
사용한 데이터: sales_100k.csv
변경 내역: 2026-07-21 최초 작성
작성자: 광주_1반_서혁인
"""

import sys
import pandas as pd
import polars as pl
import duckdb
import timeit
import logging
import os
import numpy as np

# 로거 설정 (오류 및 상태 메시지 출력용)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    file_path = "sales_100k.csv"

    # [Data Setup] 별도의 외부 입력 파일명이 존재할 경우 터미널 실행시 함께 입력
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # [Data Setup] 파일이 없을 경우 이상치가 포함된 임의의 10만 건 테스트 데이터 생성
    if not os.path.exists(file_path):
        logger.warning(f"'{file_path}' 파일이 존재하지 않아 임의의 데이터를 자동 생성합니다.")
        np.random.seed(42)
        n_rows = 100_000
        
        # 정상 데이터(99,900건) + 이상치 데이터(100건) 임의 생성
        normal_amount = np.random.normal(500, 100, n_rows - 100)
        outlier_amount = np.random.uniform(2000, 5000, 100)
        
        pd.DataFrame({
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Toys'], n_rows),
            'amount': np.append(normal_amount, outlier_amount) 
        }).to_csv(file_path, index=False)
        logger.info(f"'{file_path}' 생성 완료.\n")

    # [예외/오류 처리] 파일 읽기 및 분석 과정 전체를 try-except로 감싸 안정성 확보
    try:
        # =====================================================================
        # 1) Pandas EDA 기초 탐색 + 이상치 처리 (IQR)
        # =====================================================================
        print("="*60)
        print("📊 [1] Pandas EDA & 기초 탐색")
        
        df = pd.read_csv(file_path)
        
        print("\n[df.info() 결과]")
        df.info()
        
        print("\n[df.isnull().sum() 결과]")
        print(df.isnull().sum())
        
        # IQR 공식 적용
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 제거 전/후 행 수 계산 및 필터링 (between 사용)
        before_count = len(df)
        df_filtered = df[df['amount'].between(lower_bound, upper_bound)]
        after_count = len(df_filtered)
        
        print(f"\n[이상치 제거 결과 (IQR 방식)]")
        print(f" - 기준 범위: {lower_bound:.2f} ~ {upper_bound:.2f}")
        print(f" - 제거 전 행 수: {before_count:,}건")
        print(f" - 제거 후 행 수: {after_count:,}건")
        print(f" - 제거된 이상치: {before_count - after_count:,}건")

        # =====================================================================
        # 2) Pandas groupby named aggregation
        # =====================================================================
        print("\n" + "="*60)
        print("🐼 [2] Pandas Groupby Named Aggregation 결과")
        
        pandas_result = df_filtered.groupby(['region', 'category']).agg(
            total=('amount', 'sum'),
            mean=('amount', 'mean'),
            count=('amount', 'count')
        ).sort_values(by='total', ascending=False).reset_index()
        
        print(pandas_result.head())

        # =====================================================================
        # 3) Polars Lazy API로 동일 집계 작성
        # =====================================================================
        print("\n" + "="*60)
        print("🐻‍❄️ [3] Polars Lazy API 결과 (scan_csv 체인 완성)")
        
        # scan_csv -> filter -> group_by -> agg -> sort -> collect
        polars_result = (
            pl.scan_csv(file_path)
            .filter(pl.col('amount').is_between(lower_bound, upper_bound))
            .group_by(['region', 'category'])
            .agg(
                pl.col('amount').sum().alias('total'),
                pl.col('amount').mean().alias('mean'),
                pl.col('amount').count().alias('count')
            )
            .sort('total', descending=True)
            .collect()
        )
        print(polars_result.head(5))

        # =====================================================================
        # 4) DuckDB SQL + 세 도구 성능 비교
        # =====================================================================
        print("\n" + "="*60)
        print("🦆 [4] DuckDB SQL 집계 결과")
        
        # SQL GROUP BY로 동일 집계 작성
        query = f"""
            SELECT 
                region, 
                category, 
                SUM(amount) AS total, 
                AVG(amount) AS mean, 
                COUNT(amount) AS count
            FROM read_csv_auto('{file_path}')
            WHERE amount BETWEEN {lower_bound} AND {upper_bound}
            GROUP BY region, category
            ORDER BY total DESC
        """
        duckdb_result = duckdb.sql(query).df()
        print(duckdb_result.head())

        # ---------------------------------------------------------
        # 실행 시간 측정 (timeit 반복 횟수 3회 통일)
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("⏱️ [5] 도구별 성능 비교 (반복 횟수: 3회 평균)")
        
        iterations = 3
        
        # Pandas 실행 함수
        def run_pandas():
            temp_df = pd.read_csv(file_path)
            return temp_df[temp_df['amount'].between(lower_bound, upper_bound)] \
                    .groupby(['region', 'category']) \
                    .agg(total=('amount', 'sum'), mean=('amount', 'mean'), count=('amount', 'count')) \
                    .sort_values(by='total', ascending=False)

        # Polars 실행 함수
        def run_polars():
            return (
                pl.scan_csv(file_path)
                .filter(pl.col('amount').is_between(lower_bound, upper_bound))
                .group_by(['region', 'category'])
                .agg(
                    pl.col('amount').sum().alias('total'),
                    pl.col('amount').mean().alias('mean'),
                    pl.col('amount').count().alias('count')
                )
                .sort('total', descending=True)
                .collect()
            )

        # DuckDB 실행 함수
        def run_duckdb():
            return duckdb.sql(query).df()

        # timeit 측정 (total time / iterations)
        time_pandas = timeit.timeit(run_pandas, number=iterations) / iterations
        time_polars = timeit.timeit(run_polars, number=iterations) / iterations
        time_duckdb = timeit.timeit(run_duckdb, number=iterations) / iterations

        print(f" 1. Pandas 평균 실행 시간: {time_pandas:.4f} 초")
        print(f" 2. Polars 평균 실행 시간: {time_polars:.4f} 초")
        print(f" 3. DuckDB 평균 실행 시간: {time_duckdb:.4f} 초")
        print("="*60)

    except FileNotFoundError:
        logger.error(f"지정된 경로에 '{file_path}' 파일이 없습니다. 데이터를 준비해주세요.")
    except Exception as e:
        logger.error(f"파이프라인 실행 중 예기치 않은 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()