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

# -----------------------------
# 데이터 불러오기 및 전처리
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 필요한 열만 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 결측값 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년 이후 데이터 제외
    df = df[df["연도"] <= LAST_YEAR].copy()

    # 연도별 관측일 수와 평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[yearly["관측일수"] >= MIN_OBSERVATIONS].copy()

    # 회귀용 독립변수: 1908년부터 지난 연수
    yearly["경과연수"] = yearly["연도"] - BASE_YEAR

    return yearly.sort_values("연도").reset_index(drop=True)


df = load_data()

# -----------------------------
# 회귀 계산
# -----------------------------
x = df["경과연수"].to_numpy()
y = df["평균기온"].to_numpy()

# y = slope * x + intercept
slope, intercept = np.polyfit(x, y, 1)

# 예측값
df["회귀예측"] = slope * df["경과연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀식 표시용
sign = "+" if intercept >= 0 else "-"
regression_equation = (
    f"평균기온 = {slope:.4f} × (연도 - {BASE_YEAR}) "
    f"{sign} {abs(intercept):.4f}"
)

# -----------------------------
# 제목
# -----------------------------
st.title("🌡️ 기온 예측기")
st.caption(
    "서울 일별 평균기온 데이터를 이용해 연평균기온을 계산하고 "
    "선형회귀로 미래 기온을 추정합니다."
)

# -----------------------------
# 회귀 정보
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{len(df):,}개")

with col2:
    st.metric("시작 연도", f"{df['연도'].min()}년")

with col3:
    st.metric("끝 연도", f"{df['연도'].max()}년")

with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.info(
    f"회귀식: **{regression_equation}**  \n"
    f"조건: 2025년까지의 데이터 중 연간 관측일이 "
    f"{MIN_OBSERVATIONS}일 이상인 연도만 사용"
)

# -----------------------------
# 연도 슬라이더
# -----------------------------
selected_year = st.slider(
    "예상 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - BASE_YEAR
predicted_temp = slope * selected_x + intercept

st.subheader(f"{selected_year}년 예상 평균기온")

st.metric(
    label=f"{selected_year}년 예상 평균기온",
    value=f"{predicted_temp:.2f} °C",
)

# -----------------------------
# Plotly 산점도 + 회귀선
# -----------------------------
fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(
            size=7,
            color="#2E86DE",
            opacity=0.75,
        ),
        customdata=df[["관측일수"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        ),
    )
)

# 회귀선
line_years = np.array(
    [df["연도"].min(), df["연도"].max()]
)
line_x = line_years - BASE_YEAR
line_y = slope * line_x + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(
            color="#E74C3C",
            width=3,
        ),
        hovertemplate=(
            "%{x}년<br>"
            "회귀 예측: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

# 선택한 연도의 예측점
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker=dict(
            size=14,
            color="#F39C12",
            symbol="star",
            line=dict(
                color="white",
                width=2,
            ),
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            "예상 평균기온: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 회귀 직선",
    xaxis=dict(
        title="연도",
        tickmode="linear",
        dtick=10,
    ),
    yaxis=dict(
        title="평균기온 (°C)",
    ),
    hovermode="closest",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=60, r=30, t=90, b=60),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 데이터 확인
# -----------------------------
with st.expander("회귀에 사용된 연도별 데이터 보기"):
    display_df = df[["연도", "관측일수", "평균기온", "회귀예측"]].copy()
    display_df["평균기온"] = display_df["평균기온"].round(2)
    display_df["회귀예측"] = display_df["회귀예측"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "데이터 출처: 서울 일별 기온 데이터(seoul.csv). "
    "회귀는 관측일 300일 이상인 2025년 이하 연도만 대상으로 합니다."
)
