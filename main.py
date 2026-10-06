import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_YEAR = 2025
MIN_OBSERVATIONS = 300
TRAIN_END_YEAR = 2005


# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 결측값 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년 이후 데이터 제외
    df = df[df["연도"] <= LAST_YEAR].copy()

    # 연도별 평균기온과 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일수가 300일 미만인 연도 제외
    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATIONS
    ].copy()

    # 1908년을 기준으로 한 경과연수
    yearly["경과연수"] = (
        yearly["연도"] - BASE_YEAR
    )

    return (
        yearly
        .sort_values("연도")
        .reset_index(drop=True)
    )


df = load_data()


# -----------------------------
# 훈련용 / 테스트용 데이터 분리
# -----------------------------
# 2005년 이전 = 훈련용
train = df[df["연도"] < TRAIN_END_YEAR].copy()

# 2005년부터 = 테스트용
test = df[df["연도"] >= TRAIN_END_YEAR].copy()


# -----------------------------
# 연도 표준화
# -----------------------------
# 9차 다항식 계산에서 숫자가 너무 커지는 것을
# 막기 위해 연도를 표준화한다.
#
# 중요한 점:
# 평균과 표준편차도 훈련 데이터만 이용한다.

train_year_mean = train["연도"].mean()
train_year_std = train["연도"].std()

if train_year_std == 0:
    train_year_std = 1

train["표준화연도"] = (
    train["연도"] - train_year_mean
) / train_year_std

test["표준화연도"] = (
    test["연도"] - train_year_mean
) / train_year_std

df["표준화연도"] = (
    df["연도"] - train_year_mean
) / train_year_std
