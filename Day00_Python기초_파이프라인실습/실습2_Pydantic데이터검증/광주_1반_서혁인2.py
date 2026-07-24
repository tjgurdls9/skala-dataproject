"""
파일명: 광주_1반_서혁인_실습2.py
프로그램 설명: Pydantic을 활용한 데이터 검증 및 CSV/JSON 파일 I/O 파이프라인 실습
사용 데이터: Python_Practice2_Data.json
변경 내역: 2026-07-20 최초 작성
작성자: 광주_1반_서혁인
"""

import json
import csv
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ValidationError

# 로거 설정 (운영 환경 대비 타임스탬프 및 로그 레벨 표준화)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 2) Pydantic v2 스키마 정의
# ==========================================
class SalesRecord(BaseModel):
    """
    [Data Schema Definition]
    매장 매출 데이터 무결성 검증을 위한 Pydantic 모델.
    - 필수 필드 누락 시 ValidationError 발생 유도
    - 비즈니스 로직에 따른 제약조건(금액 > 0) 선언적 정의
    """
    month: str
    region: str
    amount: int = Field(..., gt=0)  # 매출액은 반드시 0 초과여야 함
    category: Optional[str] = None  # 카테고리는 선택 값(Optional)으로 누락 허용

# ==========================================
# 1) 예외 처리 + 파일 읽기
# ==========================================
def safe_load_csv(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    [Safe File I/O Pipeline]
    외부 CSV 파일을 안정적으로 로드하기 위한 방어적 함수.
    FileNotFoundError 및 예상치 못한 예외(Exception)를 핸들링하여 앱 크래시 방지.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # [Defensive Coding] 데이터 타입 변환 시 발생할 수 있는 캐스팅 에러 방어
                if 'amount' in row and row['amount']:
                    try:
                        row['amount'] = int(float(row['amount'])) # 소수점 형태 문자열까지 유연하게 파싱
                    except ValueError:
                        pass # 변환 실패 시 원본 유지 혹은 패스하여 파이프라인 중단 방지
                data.append(row)
        logger.info(f"'{file_path}' 로딩 성공")
        return data
    except FileNotFoundError:
        logger.error(f"'{file_path}' 완성파일 없음")
        return None
    except Exception as e:
        logger.error(f"파일 읽기 오류: {e}")
        return None
    finally:
        # [Resource Lifecycle Management] 성공/실패 여부와 관계없이 실행을 보장하는 블록
        print(f"[{file_path}] 로딩 종료") # Checkpoint: finally 누락 방어

def main():
    # [Step 1: Raw Data Ingestion] 원본 JSON 적재 단계
    try:
        with open('Python_Practice2_Data.json', 'r', encoding='utf-8') as f:
            raw_data = json.load(f)['Sales']
    except FileNotFoundError:
        print("오류: JSON 파일을 찾을 수 없습니다.")
        return

    # ==========================================
    # 3) 검증 파이프라인 (valid / errors 분리)
    # ==========================================
    valid, errors = [], []

    # [Step 2: Validation & Partitioning] ETL 파이프라인의 핵심 검증 단계
    for row in raw_data:
        try:
            record = SalesRecord(**row)
            valid.append(record.model_dump()) # Checkpoint: model_dump 사용 (Pydantic v2 표준)
        except ValidationError as e:
            error_detail = e.errors()
            print(f"[ValidationError 발생] 불량 데이터: {row}")
            print(f" -> 사유: {error_detail}\n") 
            # [Audit Trail] 추후 데이터 정제(Data Cleansing)를 위해 실패 원인과 로우 데이터 함께 적재
            errors.append({"row": row, "error": error_detail})

    # [Step 3: Monitoring & Reporting] 배치 작업 결과 요약 보고
    print("=" * 50)
    print("📊 [데이터 검증 결과 요약]")
    print(f" - 전체 데이터 : {len(raw_data)}건")
    print(f" - 정상(valid) : {len(valid)}건")
    print(f" - 불량(errors): {len(errors)}건")
    
    if len(valid) == 93 and len(errors) == 7:
        print(" 평가 기준(93건/7건) 일치 여부: [통과 ✅]")
    else:
        print(" 평가 기준(93건/7건) 일치 여부: [불통과 ❌]")
    print("=" * 50 + "\n")

    # Checkpoint: valid 93건 / errors 7건 assert (비즈니스 요구사항 정합성 검증)
    assert len(valid) == 93, f"valid 건수 불일치 ({len(valid)}건)"
    assert len(errors) == 7, f"errors 건수 불일치 ({len(errors)}건)"

    # ==========================================
    # 4) 결과 파일 저장 + 재로딩 확인
    # ==========================================
    valid_csv_path, errors_json_path = 'valid_sales.csv', 'error_sales.json'

    # [Step 4: Persistence & Sink] 정제된 데이터 및 에러 로깅 파일화
    
    # 만약 정상 데이터가 0건이더라도 스키마 기반 필드 헤더를 유지하여 후속 프로세스 크래시 방어
    fieldnames = list(SalesRecord.model_fields.keys())
    if valid and valid[0]:
        fieldnames = list(valid[0].keys())

    # ① 정상 레코드를 구조화된 CSV 포맷으로 적재
    with open(valid_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if valid:
            writer.writerows(valid)
            
    # ② 에러 로그를 상세 분석이 가능하도록 JSON 포맷으로 적재 (Checkpoint: ensure_ascii=False로 인코딩 깨짐 방지)
    with open(errors_json_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=4)

    # [Step 5: Integration Test / Smoke Test] 적재된 아티팩트의 무결성 검증
    
    # ③ 저장 직후 CSV 파일 재로딩을 통한 End-to-End 파이프라인 검증
    reloaded = safe_load_csv(valid_csv_path)
    assert reloaded is not None and len(reloaded) == 93, "CSV 재로딩 실패" # Checkpoint 통과

    # ④ 예외 케이스(Negative Test): 존재하지 않는 파일 핸들링에 대한 단언문 테스트
    assert safe_load_csv("없는파일.csv") is None, "None 반환 실패" # Checkpoint 통과

if __name__ == "__main__":
    main()