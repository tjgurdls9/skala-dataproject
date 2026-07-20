"""
파일명: test_pipeline.py
프로그램 설명: Day 1 종합 실습 - Pydantic 모델 검증 로직에 대한 pytest 단위 테스트 코드
변경 내역: 2026-07-20 최초 작성
작성자: 광주_1반_서혁인
"""

import pytest
from pydantic import ValidationError
from pipeline_main import PipelineDataRecord


def test_pipeline_data_record_valid():
    """
    [Test: Valid Data]
    정상적인 타입의 데이터가 입력되었을 때 Pydantic 모델이 올바르게 파싱 및 생성되는지 검증
    """
    valid_data = {
        "latitude": 37.5665,
        "longitude": 126.9780,
        "timezone": "Asia/Seoul",
        "country_name": "South Korea",
        "ip_query": "8.8.8.8"
    }
    record = PipelineDataRecord(**valid_data)
    assert record.latitude == 37.5665


def test_pipeline_data_record_invalid():
    """
    [Test: Invalid Data & ValidationError]
    잘못된 타입(문자열 등)의 데이터가 위도(latitude) 필드에 입력되었을 때,
    Pydantic ValidationError 예외가 정상적으로 발생하는지 검증
    """
    invalid_data = {
        "latitude": "문자열오류",
        "longitude": 126.9780,
        "timezone": "Asia/Seoul",
        "country_name": "South Korea",
        "ip_query": "8.8.8.8"
    }
    with pytest.raises(ValidationError):
        PipelineDataRecord(**invalid_data)