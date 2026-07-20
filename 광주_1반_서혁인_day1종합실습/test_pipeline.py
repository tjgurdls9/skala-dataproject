import pytest
from pydantic import ValidationError
from pipeline_main import PipelineDataRecord

def test_pipeline_data_record_valid():
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
    invalid_data = {
        "latitude": "문자열오류",
        "longitude": 126.9780,
        "timezone": "Asia/Seoul",
        "country_name": "South Korea",
        "ip_query": "8.8.8.8"
    }
    with pytest.raises(ValidationError):
        PipelineDataRecord(**invalid_data)