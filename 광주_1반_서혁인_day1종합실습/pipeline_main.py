"""
파일명: pipeline_main.py
프로그램 설명: Day 1 종합 실습 - 비동기 API 수집, Pydantic 검증, CSV/Parquet 저장 및 성능 비교 파이프라인
변경 내역: 2026-07-20 최초 작성
작성자: 광주_1반_서혁인
"""

import asyncio
import time
import logging
from typing import Dict, Any
import httpx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

# 로거 설정 (운영 환경 표준 포맷)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# API 엔드포인트 정의
API_ENDPOINTS = {
    "weather": "https://api.open-meteo.com/v1/forecast?latitude=37.5665&longitude=126.9780&hourly=temperature_2m,precipitation_probability&forecast_days=3&timezone=Asia/Seoul",
    "country": "https://countries.dev/alpha/KOR",
    "ip_info": "http://ip-api.com/json/8.8.8.8"
}


# ==========================================
# 1) Pydantic v2 스키마 정의 (데이터 검증)
# ==========================================
class PipelineDataRecord(BaseModel):
    """
    [Data Schema Definition]
    수집된 이중/복합 구조 데이터의 무결성 및 타입 검증을 위한 모델.
    """
    latitude: float = Field(..., description="위도")
    longitude: float = Field(..., description="경도")
    timezone: str = Field(..., description="시간대")
    country_name: str = Field(..., description="국가 이름")
    ip_query: str = Field(..., description="조회된 IP 주소")


# ==========================================
# 2) 비동기 API 수집 함수
# ==========================================
async def fetch_api(client: httpx.AsyncClient, name: str, url: str) -> tuple[str, Any]:
    """
    [Async HTTP Client]
    단일 API에 비동기 요청을 보내고 JSON 응답을 반환하는 함수.
    """
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        logger.info(f"[{name}] API 수집 성공")
        return name, response.json()
    except Exception as e:
        logger.error(f"[{name}] API 수집 실패: {e}")
        return name, None


async def collect_all_apis() -> Dict[str, Any]:
    """
    [Async Gather Pipeline]
    asyncio.gather()를 활용하여 3개 API를 동시에 수집하는 파이프라인.
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_api(client, "weather", API_ENDPOINTS["weather"]),
            fetch_api(client, "country", API_ENDPOINTS["country"]),
            fetch_api(client, "ip_info", API_ENDPOINTS["ip_info"])
        ]
        results = await asyncio.gather(*tasks)
        return {name: data for name, data in results if data is not None}


# ==========================================
# 3) 메인 프로세스 (수집 -> 검증 -> 저장 및 성능 비교)
# ==========================================
def main():
    # 1. 비동기 수집 실행
    start_time = time.time()
    raw_data = asyncio.run(collect_all_apis())
    logger.info(f"비동기 수집 소요 시간: {time.time() - start_time:.4f}초")

    if len(raw_data) < 3:
        logger.error("일부 API 수집 실패로 파이프라인을 중단합니다.")
        return

    # 2. 데이터 가공 및 파싱 (각 API 응답 구조 매핑)
    try:
        weather_json = raw_data.get("weather", {})
        country_json = raw_data.get("country", {})
        ip_json = raw_data.get("ip_info", {})

        # Countries.dev 구조 대응 (data 키 존재 여부 방어)
        country_data = country_json.get("data", country_json)

        parsed_payload = {
            "latitude": weather_json.get("latitude"),
            "longitude": weather_json.get("longitude"),
            "timezone": weather_json.get("timezone"),
            "country_name": country_data.get("name", "Unknown"),
            "ip_query": ip_json.get("query", "Unknown")
        }
    except Exception as e:
        logger.error(f"데이터 파싱 중 오류 발생: {e}")
        return

    # 3. Pydantic 스키마 검증
    try:
        validated_record = PipelineDataRecord(**parsed_payload)
        print("\n[Pydantic 스키마 검증 결과] 통과 ✅")
        print(validated_record.model_dump_json(indent=2))
    except ValidationError as e:
        print(f"[ValidationError 발생] 데이터 구조 불일치: {e}")
        return

    # Pandas DataFrame 변환 (저장 테스트용)
    df = pd.DataFrame([validated_record.model_dump()])

    # 4. 저장 및 성능 비교 (CSV vs Parquet)
    csv_path = "pipeline_result.csv"
    parquet_path = "pipeline_result.parquet"

    # --- CSV 쓰기 성능 측정 ---
    t0 = time.time()
    df.to_csv(csv_path, index=False, encoding='utf-8')
    csv_write_time = time.time() - t0

    # --- Parquet 쓰기 성능 측정 ---
    t0 = time.time()
    df.to_parquet(parquet_path, index=False)
    parquet_write_time = time.time() - t0

    # --- CSV 읽기 성능 측정 ---
    t0 = time.time()
    _ = pd.read_csv(csv_path)
    csv_read_time = time.time() - t0

    # --- Parquet 읽기 성능 측정 ---
    t0 = time.time()
    _ = pd.read_parquet(parquet_path)
    parquet_read_time = time.time() - t0

    # 성능 비교 결과 출력 (포맷 구문 수정 완료)
    print("\n" + "=" * 50)
    print("📊 [CSV vs Parquet 성능 비교 결과]")
    print(f" - CSV     | 쓰기: {csv_write_time:.6f}초 | 읽기: {csv_read_time:.6f}초")
    print(f" - Parquet | 쓰기: {parquet_write_time:.6f}초 | 읽기: {parquet_read_time:.6f}초")
    print("=" * 50)


if __name__ == "__main__":
    main()