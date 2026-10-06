```python
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

    # 평균기온 숫자 변환
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

    # 2025년 이후 제외
    df = df[df["연도"] <= LAST_YEAR].copy()

    # 연도별 평균기온 및 관측일수
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

    # 기본 회귀용 경과연수
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
# 훈련 / 테스트 데이터 분리
# -----------------------------
train = df[df["연도"] < TRAIN_END_YEAR].copy()
test = df[df["연도"] >= TRAIN_END_YEAR].copy()

# -----------------------------
# 9차 곡선의 수치 불안정 방지
# -----------------------------
# 연도 자체를 사용하지 않고
# 훈련 데이터의 평균과 표준편차를 이용해
# 표준화된 연도를 사용한다.

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

# 전체 데이터에도 같은 변환 적용
df["표준화연도"] = (
    df["연도"] - train_year_mean
) / train_year_std

# -----------------------------
# 다항회귀 함수
# -----------------------------
def make_polynomial_model(degree):
    """
    훈련 데이터만 이용하여 degree차 다항식을 학습
    """
    coefficients = np.polyfit(
        train["표준화연도"].to_numpy(),
        train["평균기온"].to_numpy(),
        degree,
    )

    return np.poly1d(coefficients)


# -----------------------------
# 1차 / 3차 / 9차 모델 학습
# -----------------------------
models = {}

for degree in [1, 3, 9]:
    models[degree] = make_polynomial_model(degree)


# -----------------------------
# 테스트 데이터로 채점
# -----------------------------
test_results = []

for degree, model in models.items():

    test_x = test["표준화연도"].to_numpy()
    test_y = test["평균기온"].to_numpy()

    # 테스트 데이터 예측
    test_prediction = model(test_x)

    # 평균 절대 오차(MAE)
    mae = np.mean(
        np.abs(test_y - test_prediction)
    )

    # 2050년 예측
    x_2050 = (
        2050 - train_year_mean
    ) / train_year_std

    prediction_2050 = model(x_2050)

    test_results.append(
        {
            "모델": f"{degree}차 곡선",
            "테스트 평균 오차(MAE)": mae,
            "2050년 예측값(°C)": prediction_2050,
        }
    )


result_df = pd.DataFrame(test_results)

# -----------------------------
# 제목
# -----------------------------
st.title("🌡️ 기온 예측기")

st.caption(
    "서울 일별 평균기온 데이터를 이용해 연평균기온을 계산하고 "
    "1차·3차·9차 다항회귀로 미래 기온을 추정합니다."
)

# -----------------------------
# 훈련 / 테스트 데이터 정보
# -----------------------------
st.subheader("📚 훈련용 / 테스트용 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train):,}개"
    )

    st.write(
        f"{train['연도'].min()}년 ~ "
        f"{train['연도'].max()}년"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test):,}개"
    )

    st.write(
        f"{test['연도'].min()}년 ~ "
        f"{test['연도'].max()}년"
    )

st.info(
    "훈련용 데이터는 2005년 이전, "
    "테스트용 데이터는 2005년부터 사용합니다. "
    "세 모델 모두 테스트 데이터를 학습에 사용하지 않았습니다."
)

# -----------------------------
# 모델 평가 결과
# -----------------------------
st.subheader("📊 모델별 테스트 결과")

display_results = result_df.copy()

display_results["테스트 평균 오차(MAE)"] = (
    display_results["테스트 평균 오차(MAE)"]
    .round(3)
)

display_results["2050년 예측값(°C)"] = (
    display_results["2050년 예측값(°C)"]
    .round(2)
)

st.table(display_results)

st.caption(
    "MAE는 테스트 데이터에서 실제 연평균기온과 "
    "예측값의 차이를 절댓값으로 계산한 평균입니다. "
    "값이 작을수록 테스트 데이터에서 더 잘 예측한 모델입니다."
)

# -----------------------------
# 가장 좋은 모델
# -----------------------------
best_model = result_df.loc[
    result_df["테스트 평균 오차(MAE)"].idxmin()
]

st.success(
    f"테스트 데이터에서 가장 오차가 작은 모델: "
    f"**{best_model['모델']}** "
    f"(MAE: {best_model['테스트 평균 오차(MAE)']:.3f} °C)"
)

# -----------------------------
# 2050년 예측
# -----------------------------
st.subheader("🔮 2050년 예측")

pred_cols = st.columns(3)

for i, row in result_df.iterrows():
    with pred_cols[i]:
        st.metric(
            row["모델"],
            f"{row['2050년 예측값(°C)']:.2f} °C"
        )

# -----------------------------
# 연도 슬라이더
# -----------------------------
st.subheader("📅 원하는 연도의 예상기온")

selected_year = st.slider(
    "예상 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = (
    selected_year - train_year_mean
) / train_year_std

# 세 모델의 예측값
selected_predictions = {}

for degree, model in models.items():
    selected_predictions[degree] = model(
        selected_x
    )

st.write(
    f"**{selected_year}년 모델별 예상 평균기온**"
)

prediction_cols = st.columns(3)

for i, degree in enumerate([1, 3, 9]):
    with prediction_cols[i]:
        st.metric(
            f"{degree}차 곡선",
            f"{selected_predictions[degree]:.2f} °C"
        )

# -----------------------------
# Plotly 그래프
# -----------------------------
fig = go.Figure()

# 실제 연평균기온
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

# -----------------------------
# 회귀곡선 그리기
# -----------------------------
line_years = np.arange(
    df["연도"].min(),
    2051
)

line_standardized = (
    line_years - train_year_mean
) / train_year_std

curve_colors = {
    1: "#E74C3C",
    3: "#27AE60",
    9: "#8E44AD",
}

for degree, model in models.items():

    line_y = model(line_standardized)

    fig.add_trace(
        go.Scatter(
            x=line_years,
            y=line_y,
            mode="lines",
            name=f"{degree}차 곡선",
            line=dict(
                color=curve_colors[degree],
                width=3,
            ),
            hovertemplate=(
                f"{degree}차 곡선<br>"
                "%{x}년<br>"
                "예측: %{y:.2f} °C"
                "<extra></extra>"
            ),
        )
    )

# -----------------------------
# 선택한 연도 예측점
# -----------------------------
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_predictions[1]],
        mode="markers",
        name=f"{selected_year}년 1차 예측",
        marker=dict(
            size=12,
            color="#F39C12",
            symbol="star",
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            "1차 예상: %{y:.2f} °C"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 다항회귀 곡선",
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
    margin=dict(
        l=60,
        r=30,
        t=90,
        b=60,
    ),
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# 데이터 확인
# -----------------------------
with st.expander("훈련용 데이터 보기"):

    train_display = train[
        ["연도", "관측일수", "평균기온"]
    ].copy()

    train_display["평균기온"] = (
        train_display["평균기온"]
        .round(2)
    )

    st.dataframe(
        train_display,
        use_container_width=True,
        hide_index=True,
    )


with st.expander("테스트용 데이터와 모델 예측 보기"):

    test_display = test[
        ["연도", "관측일수", "평균기온"]
    ].copy()

    # 각 모델의 테스트 예측값
    test_x = test["표준화연도"].to_numpy()

    test_display["1차 예측"] = (
        models[1](test_x)
    )

    test_display["3차 예측"] = (
        models[3](test_x)
    )

    test_display["9차 예측"] = (
        models[9](test_x)
    )

    test_display = test_display.round(2)

    st.dataframe(
        test_display,
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------
# 분석 방법 설명
# -----------------------------
st.subheader("🔎 분석 방법")

st.write(
    """
**1. 데이터 분리**

- 2005년 이전 → 훈련용
- 2005년부터 → 테스트용

**2. 모델 학습**

- 1차 다항식
- 3차 다항식
- 9차 다항식

세 모델 모두 **훈련용 데이터만 이용하여 학습**했습니다.

**3. 모델 평가**

학습에 사용하지 않은 **테스트용 데이터**를 이용해
평균 절대 오차(MAE)를 계산했습니다.

**4. 2050년 예측**

테스트 데이터를 다시 학습에 넣지 않고,
훈련용 데이터로 만들어진 각 모델에 2050년을 입력했습니다.

**5. 고차 곡선 안정화**

연도를 그대로 1908, 1950, 2000처럼 사용하지 않고,
훈련 데이터의 평균과 표준편차를 이용해 표준화한 뒤
다항회귀를 계산했습니다.
"""
)

st.caption(
    "데이터 출처: 서울 일별 기온 데이터(seoul.csv). "
    "연간 관측일수가 300일 이상인 연도만 사용합니다."
)
```
