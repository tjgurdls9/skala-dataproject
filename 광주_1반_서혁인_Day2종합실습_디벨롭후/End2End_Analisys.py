"""
파일명: 광주_1반_5조_day2종합실습.py
프로그램 설명: [Day 2 종합 실습] Adult Census Income 데이터 활용 End2End 분석 프로젝트
프로그램 비고:
    - Pandas/Polars 양방향 로딩 및 결측(?)·중복 처리, 기본 EDA
    - 기술통계·상관분석·t-test·선형회귀 기반 인사이트 도출
    - 4종 시각화(교육-소득 / 교육-근무시간, Seaborn+Plotly 이중축)
    - 분석 결과를 읽기 쉬운 report.md 로 자동 생성
작성자: 광주_1반_5조
"""

import os
import urllib.request
import joblib
import logging
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# sklearn 머신러닝 및 파이프라인 관련 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 한글 폰트 설정
if os.name == 'posix':
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="Malgun Gothic")
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_step(step_num, title):
    print("\n" + "="*75)
    print(f" 🚀 [STEP {step_num}] {title}")
    print("="*75)

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    try:
        print_step(1, "데이터 로드 및 정제 (Pandas vs Polars 양방향 비교)")

        # 원본을 1회 다운로드해 두 라이브러리 공통 입력으로 사용
        raw_path = "adult_raw.data"
        if not os.path.exists(raw_path):
            urllib.request.urlretrieve(url, raw_path)

        # (1) Pandas 로딩 — '?' 를 결측으로 인식 (na_values="?" 로 정확히 지정)
        df_pd = pd.read_csv(raw_path, header=None, names=cols, na_values="?", skipinitialspace=True)
        for c in ['age', 'education-num', 'hours-per-week']:
            df_pd[c] = pd.to_numeric(df_pd[c], errors='coerce')
        df_pd['income'] = df_pd['income'].astype(str).str.strip()
        initial_shape = df_pd.shape
        missing_count = int(df_pd.isnull().sum().sum())

        # (2) Polars 로딩 — 독립적으로 raw 로딩 후 문자열 공백/빈 행 정리
        df_pl = pl.read_csv(raw_path, has_header=False, new_columns=cols, null_values=" ?")
        _str = [c for c, dt in zip(df_pl.columns, df_pl.dtypes) if dt == pl.String]
        df_pl = df_pl.with_columns([pl.col(c).str.strip_chars() for c in _str]) \
                     .filter(~pl.all_horizontal(pl.all().is_null()))

        # (3) 두 라이브러리 로딩 결과 비교
        shape_match = df_pd.shape == (df_pl.height, df_pl.width)
        logger.info(f" [+] Pandas {df_pd.shape} vs Polars {(df_pl.height, df_pl.width)} | shape 일치: {shape_match}")
        logger.info(f" [+] 결측치 인식: {missing_count:,}건 (workclass/occupation/native-country)")

        # (4) 결측·중복 처리 (결측 포함 행 제거 + 완전 중복 제거)
        df_pd = df_pd.dropna().drop_duplicates()
        dropped_count = initial_shape[0] - df_pd.shape[0]
        assert df_pd.isnull().sum().sum() == 0, "결측치 정제 실패"
        cleaned_shape = df_pd.shape  # 파생변수 추가 전(원본 15컬럼) 기준 정제 결과
        logger.info(f" [+] 정제 완료 | 결측·중복 {dropped_count:,}건 제거 -> {df_pd.shape[0]:,}행 확정")

        print_step(2, "기술통계 · 상관분석 · 통계 검정")

        # 0) 기술통계 (평균·표준편차·분위수)
        desc_stats = df_pd[['age', 'education-num', 'hours-per-week',
                            'capital-gain', 'capital-loss']].describe().round(2)
        logger.info("\n[기술통계]\n" + desc_stats.to_string())

        # 1) 타겟 수치화
        df_pd['income_num'] = df_pd['income'].apply(lambda x: 1 if '>50K' in x else 0)
        
        # 2) 상관계수
        edu_income_corr = df_pd['education-num'].corr(df_pd['income_num'])
        edu_hours_corr = df_pd['education-num'].corr(df_pd['hours-per-week'])
        
        # 3) T-test
        low_income = df_pd[df_pd['income_num'] == 0]['education-num']
        high_income = df_pd[df_pd['income_num'] == 1]['education-num']
        t_stat, p_val = stats.ttest_ind(low_income, high_income, equal_var=False)
        p_disp = f"{p_val:.2e}" if p_val > 0 else "< 1e-300 (극소값)"
        sig_txt = "유의미한 차이가 있다" if p_val < 0.05 else "유의미하지 않다"

        # 4) [추가] 리포트를 위한 선형 회귀 분석 (교육 1년당 고소득 진입 확률 상승분 도출)
        df_agg = df_pd.groupby('education-num').agg(
            count=('income', 'size'),                  
            income_rate=('income_num', 'mean'),        
            avg_hours=('hours-per-week', 'mean')       
        ).reset_index()
        df_agg['income_rate_pct'] = df_agg['income_rate'] * 100 
        
        # 교육 연수(X)에 따른 고소득 비율(Y) 회귀분석 기울기 산출
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(df_agg['education-num'], df_agg['income_rate_pct'])
        marginal_increase = slope # 교육 1년 증가당 고소득 확률 %p 증가분

        logger.info(f" [-] 교육-소득 상관계수: {edu_income_corr:.4f}")
        logger.info(f" [-] 교육-근무 상관계수: {edu_hours_corr:.4f}")
        logger.info(f" [-] 한계 상승률(Slope): 교육 1년 추가시 고소득 진입 확률 평균 {marginal_increase:.2f}%p 상승")


        print_step(3, "이중축(Dual-axis) 시각화 4종 렌더링 (Seaborn & Plotly)")

        # ==========================================
        # [주제 1] 교육 연수 vs 소득 수준 (인구수 & 고소득자 비율)
        # ==========================================
        # 1-1. Seaborn 정적 차트
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.bar(df_agg['education-num'], df_agg['count'], color='#e0e0e0', label='인구수(볼륨)')
        ax1.set_xlabel('교육 연수 (년)', fontsize=12)
        ax1.set_ylabel('데이터 볼륨 (인원 수)', fontsize=12, color='gray')
        
        ax2 = ax1.twinx()
        ax2.plot(df_agg['education-num'], df_agg['income_rate_pct'], color='#d62728', marker='o', linewidth=2.5, label='고소득자 비율')
        ax2.set_ylabel('고소득자(>50K) 비율 (%)', fontsize=12, color='#d62728')
        
        plt.title('[Seaborn] 인적 자본 투자(교육)에 따른 고소득 진입 추이', fontsize=15)
        plt.tight_layout()
        plt.savefig('seaborn_edu_vs_income.png', dpi=300)
        plt.close()
        logger.info(" [1/4] seaborn_edu_vs_income.png 저장 완료")

        # 1-2. Plotly 동적 차트
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=df_agg['education-num'], y=df_agg['count'], name="인원 수 (명)", marker_color='#e0e0e0'), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df_agg['education-num'], y=df_agg['income_rate_pct'], name="고소득 비율 (%)", mode='lines+markers', line=dict(color='#d62728', width=3)), secondary_y=True)
        fig2.update_layout(title_text='[Plotly] 인적 자본 투자(교육)에 따른 고소득 진입 추이', title_x=0.5)
        fig2.update_xaxes(title_text="교육 연수 (년)")
        fig2.update_yaxes(title_text="데이터 볼륨 (명)", secondary_y=False)
        fig2.update_yaxes(title_text="고소득자 비율 (%)", secondary_y=True)
        fig2.write_html('plotly_edu_vs_income.html')
        logger.info(" [2/4] plotly_edu_vs_income.html 저장 완료")

        # ==========================================
        # [주제 2] 교육 연수 vs 근무 시간 (인구수 & 평균 근무시간)
        # ==========================================
        # 2-1. Seaborn 정적 차트
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.bar(df_agg['education-num'], df_agg['count'], color='#e0e0e0')
        ax3.set_xlabel('교육 연수 (년)', fontsize=12)
        ax3.set_ylabel('데이터 볼륨 (인원 수)', fontsize=12, color='gray')
        
        ax4 = ax3.twinx()
        ax4.plot(df_agg['education-num'], df_agg['avg_hours'], color='#1f77b4', marker='s', linewidth=2.5)
        ax4.set_ylabel('평균 주당 근무시간 (시간)', fontsize=12, color='#1f77b4')
        
        plt.title('[Seaborn] 학력 수준에 따른 평균 근로 시간 한계 추이', fontsize=15)
        plt.tight_layout()
        plt.savefig('seaborn_edu_vs_hours.png', dpi=300)
        plt.close()
        logger.info(" [3/4] seaborn_edu_vs_hours.png 저장 완료")

        # 2-2. Plotly 동적 차트
        fig4 = make_subplots(specs=[[{"secondary_y": True}]])
        fig4.add_trace(go.Bar(x=df_agg['education-num'], y=df_agg['count'], name="인원 수 (명)", marker_color='#e0e0e0'), secondary_y=False)
        fig4.add_trace(go.Scatter(x=df_agg['education-num'], y=df_agg['avg_hours'], name="평균 근무시간 (시간)", mode='lines+markers', line=dict(color='#1f77b4', width=3)), secondary_y=True)
        fig4.update_layout(title_text='[Plotly] 학력 수준에 따른 평균 근로 시간 한계 추이', title_x=0.5)
        fig4.update_xaxes(title_text="교육 연수 (년)")
        fig4.update_yaxes(title_text="데이터 볼륨 (명)", secondary_y=False)
        fig4.update_yaxes(title_text="평균 주당 근무시간 (시간)", secondary_y=True)
        fig4.write_html('plotly_edu_vs_hours.html')
        logger.info(" [4/4] plotly_edu_vs_hours.html 저장 완료")


        print_step(4, "머신러닝 Pipeline 모델 학습 및 저장")
        
        features = ['age', 'education-num', 'hours-per-week', 'workclass', 'sex']
        X = df_pd[features]
        y = df_pd['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), ['age', 'education-num', 'hours-per-week']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['workclass', 'sex'])
        ])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info(f" [+] Model Accuracy : {acc:.4f}")
        logger.info(f" [+] Model F1-Score(weighted) : {f1:.4f}")
        joblib.dump(pipeline, "adult_income_pipeline.joblib")


        print_step(5, "분석 리포트(report.md) 자동 생성")
        
        report_content = f"""# Adult Census Income 데이터 분석 리포트

**발행일**: 2026-07-21
**작성자**: 광주_1반_5조

---

## 1. 분석 개요
- **데이터**: Adult Census Income (미국 인구조사 소득 데이터)
- **분석 목적**: 학력(교육 연수)과 근로 시간이 고소득(>50K) 달성에 미치는 영향을 분석하고, 소득 예측 모델을 구축한다.
- **분석 표본**: 결측·중복 처리 후 **{df_pd.shape[0]:,}행**

## 2. 데이터 준비 및 전처리
- **로딩 비교**: Pandas와 Polars 양쪽으로 로딩하여 shape 일치를 확인했다. (원본 {initial_shape}, 정제 후 {cleaned_shape})
- **결측치 처리**: 원본에서 `?`로 표기된 결측 **{missing_count:,}건**(workclass·occupation·native-country)을 인식하여 해당 행을 제거했다.
- **중복 처리**: 완전 중복 행을 제거했다. (결측·중복 합계 **{dropped_count:,}행** 제거)

## 3. 기술통계 (평균·표준편차·분위수)
```
{desc_stats.to_string()}
```

## 4. 상관관계 분석
- **교육 연수 ↔ 고소득 여부**: 상관계수 **{edu_income_corr:.4f}** — 뚜렷한 양의 상관관계
- **교육 연수 ↔ 근무 시간**: 상관계수 **{edu_hours_corr:.4f}** — 상대적으로 약한 상관관계

## 5. 통계 검정 (t-test)
- **검정 대상**: 저소득(≤50K) vs 고소득(>50K) 집단의 **교육 연수(education-num)** 평균 차이
- **결과**: t-통계량 = {t_stat:.4f}, p-value = {p_disp}
- **해석**: p-value가 0.05보다 작으므로, 두 집단의 교육 연수 평균 차이는 **통계적으로 {sig_txt}**. 즉 고소득 집단의 학력이 유의미하게 높다.

## 6. 핵심 인사이트
- **교육의 효과**: 선형회귀 분석 결과, 교육 연수가 **1년 증가할 때마다 고소득(>50K) 진입 비율이 평균 {marginal_increase:.2f}%p 상승**한다.
- **소득의 성격**: 교육-소득 상관은 강한 반면 교육-근무시간 상관은 약하다. 이는 고소득이 '근무 시간의 양'보다 '학력에 따른 노동의 질'에 더 좌우됨을 시사한다.

## 7. 예측 모델 (ML Pipeline)
- **구성**: `ColumnTransformer`(수치 표준화 + 범주 원핫 인코딩) + `RandomForestClassifier` 를 하나의 `Pipeline` 으로 구성
- **정확도(Accuracy)**: **{acc:.4f}**
- **F1-Score(weighted)**: **{f1:.4f}** (클래스 불균형을 반영한 지표)
- **모델 저장**: `adult_income_pipeline.joblib`

## 8. 시각화 산출물
데이터 볼륨(막대)과 추이(꺾은선)를 함께 보여주는 이중축 차트 4종:
1. `seaborn_edu_vs_income.png` — 교육 연수별 고소득 비율 (정적)
2. `plotly_edu_vs_income.html` — 교육 연수별 고소득 비율 (인터랙티브)
3. `seaborn_edu_vs_hours.png` — 학력별 평균 근무시간 (정적)
4. `plotly_edu_vs_hours.html` — 학력별 평균 근무시간 (인터랙티브)
"""

        with open("report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(" [+] 'report.md' 자동 생성 완료")
        print("\n" + "="*75)
        print(" ✅ [SUCCESS] 전체 분석 파이프라인이 완료되었습니다.")
        print("="*75 + "\n")

    except AssertionError as ae:
        logger.error(f"\n [CRITICAL ERROR] 무결성 검증 실패: {ae}")
    except Exception as e:
        logger.error(f"\n [ERROR] 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()