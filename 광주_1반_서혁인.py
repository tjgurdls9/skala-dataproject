"""
파일명: 광주_1반_서혁인.py
프로그램 설명: Python_Practice1_Data.json 데이터를 활용한 자료구조 집계 및 제너레이터 실습
사용 데이터: Python_Practice1_Data.json
사용 데이터 비고: 원본 데이터 로딩 에러(JSON 문법 오류 및 키 누락) 트러블슈팅 반영. 단순 리스트([...]) 형태였던 원본 데이터를 파이썬 코드 요구사항에 맞춰 최상위에 'Sales' 키를 갖는 올바른 JSON 객체 구조({"Sales": [...]})로 래핑하여 활용함.
작성자: 광주_1반_서혁인
"""

import json
import sys
from collections import Counter, defaultdict

def main():
    """
    메인 함수: JSON 데이터를 읽어와서 요구사항 1~4번을 수행합니다.
    """
    try:
        # 데이터 로드
        with open('Python_Practice1_Data.json', 'r', encoding='utf-8') as f:
            sales = json.load(f)['Sales'] 

        # ==========================================
        # 1) 리스트/딕셔너리 컴프리헨션
        # ==========================================
        # ① amount >= 1000인 거래만 필터링 
        filtered_sales = [sale for sale in sales if sale['amount'] >= 1000]
        
        # ② 지역별 총매출 dict 계산
        regions = {sale['region'] for sale in filtered_sales}
        region_total = {r: sum(s['amount'] for s in filtered_sales if s['region'] == r) for r in regions}
        
        # Checkpoint: region_total 값 확인
        assert region_total, "region_total 값이 비어있습니다!"
        print("1) 지역별 총매출 (amount >= 1000):", region_total)


        # ==========================================
        # 2) Counter + defaultdict
        # ==========================================
        # ① Counter로 지역별 거래 건수 카운팅
        region_counts = Counter(sale['region'] for sale in sales)


        # Counter.most_common() 순서
        print("\n2-1) 지역별 거래 건수 Top:", region_counts.most_common())

        # ② defaultdict로 카테고리별 amount 리스트
        category_amount = defaultdict(list)
        for sale in sales:
            category_amount[sale['category']].append(sale['amount'])
        print("2-2) 카테고리별 amount 리스트 예시(키 목록):", list(category_amount.keys()))


        # ==========================================
        # 3) 제너레이터 — 메모리 비교
        # ==========================================
        # 제너레이터 vs 리스트 메모리 비교
        gen_sales = (sale for sale in sales if sale['amount'] > 1000)
        list_sales = [sale for sale in sales if sale['amount'] > 1000]

        # generator sys.gensizeof < list 확인
        gen_size, list_size = sys.getsizeof(gen_sales), sys.getsizeof(list_sales)
        print(f"\n3) 메모리 비교: 제너레이터({gen_size} bytes) vs 리스트({list_size} bytes)")
        assert gen_size < list_size, "제너레이터의 메모리 사용량이 더 커야 합니다."


        # ==========================================
        # 4) 종합 - 월별 카테고리 매출 집계
        # ==========================================
        # 월별 카테고리 총매출 집계
        monthly_category_total = defaultdict(lambda: defaultdict(int))
        for sale in sales:
            monthly_category_total[sale['month']][sale['category']] += sale['amount']
        print("\n4) 월별 카테고리 매출 집계 완료.")
        
        # Checkpoint: Top3 금액 내림차순 정렬
        category_total = Counter()
        for sale in sales:
            category_total[sale['category']] += sale['amount']
            
        print("   전체 카테고리 매출 Top 3:", category_total.most_common(3))
        
    # 오류/예외 처리
    except FileNotFoundError:
        print("오류: 'Python_Practice1_Data.json' 파일을 찾을 수 없습니다.")
    except KeyError as e:
        print(f"오류: JSON 데이터 형식이 예상과 다릅니다. 키 누락: {e}")
    except Exception as e:
        print(f"알 수 없는 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()