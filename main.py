import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="서울 연평균기온 예측",
    page_icon="🌡️",
    layout="wide"
)

# ==========================================
# 데이터 설정
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

LAST_YEAR = 2025
TRAIN_END_YEAR = 2005
MIN_OBSERVATIONS = 300


# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜와 기온을 숫자/날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 데이터가 없는 행 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 2025년까지 사용
    df = df[
        df["연도"] <= LAST_YEAR
    ].copy()

    # 연도별 평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(
            평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 1년 동안 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[
        yearly["관측일수"] >= MIN_OBSERVATIONS
    ].copy()

    return yearly.sort_values(
        "연도"
    ).reset_index(drop=True)


df = load_data()


# ==========================================
# 훈련용 / 테스트용 데이터 분리
# ==========================================

# 2005년 이전 → 훈련용
train = df[
    df["연도"] < TRAIN_END_YEAR
].copy()

# 2005년부터 → 테스트용
test = df[
    df["연도"] >= TRAIN_END_YEAR
].copy()


# ==========================================
# 연도 숫자 변환
# ==========================================
# 1900, 2000 같은 큰 숫자를 그대로
# 9차식 계산에 사용하면 불안정할 수 있으므로
# 훈련 데이터의 평균과 표준편차를 이용하여
# 작은 숫자로 변환한다.

year_mean = train["연도"].mean()
year_std = train["연도"].std()

if year_std == 0:
    year_std = 1

train["변환연도"] = (
    train["연도"] - year_mean
) / year_std

test["변환연도"] = (
    test["연도"] - year_mean
) / year_std


# ==========================================
# 다항회귀 모델
# ==========================================

def make_model(degree):

    x = train["변환연도"].to_numpy()
    y = train["평균기온"].to_numpy()

    # 1차 / 3차 / 9차 다항식
    # 훈련 데이터만 사용한다.
    model = np.polynomial.Polynomial.fit(
        x,
        y,
        degree
    )

    return model


model_1 = make_model(1)
model_3 = make_model(3)
model_9 = make_model(9)


# ==========================================
# 테스트 데이터 예측
# ==========================================

test_x = test["변환연도"].to_numpy()
test_y = test["평균기온"].to_numpy()

prediction_1 = model_1(test_x)
prediction_3 = model_3(test_x)
prediction_9 = model_9(test_x)


# ==========================================
# 테스트 평균 오차(MAE)
# ==========================================
# 실제값과 예측값의 차이를 절댓값으로 만든 후
# 평균을 계산한다.

mae_1 = np.mean(
    np.abs(test_y - prediction_1)
)

mae_3 = np.mean(
    np.abs(test_y - prediction_3)
)

mae_9 = np.mean(
    np.abs(test_y - prediction_9)
)


# ==========================================
# 2050년 예측
# ==========================================

year_2050 = (
    2050 - year_mean
) / year_std

prediction_2050_1 = float(
    model_1(year_2050)
)

prediction_2050_3 = float(
    model_3(year_2050)
)

prediction_2050_9 = float(
    model_9(year_2050)
)


# ==========================================
# 결과표
# ==========================================

result_df = pd.DataFrame({

    "모델": [
        "1차 (직선)",
        "3차 곡선",
        "9차 곡선"
    ],

    "테스트 평균 오차 (MAE, ℃)": [
        mae_1,
        mae_3,
        mae_9
    ],

    "2050년 예측값 (℃)": [
        prediction_2050_1,
        prediction_2050_3,
        prediction_2050_9
    ]
})


# ==========================================
# 화면 제목
# ==========================================

st.title(
    "🌡️ 서울 연평균기온 예측"
)

st.write(
    "2005년 이전 데이터를 훈련용으로 사용하고, "
    "2005년부터의 데이터를 학습에 사용하지 않은 "
    "테스트용 데이터로 사용하여 "
    "1차·3차·9차 곡선을 비교합니다."
)


# ==========================================
# 훈련 / 테스트 연도 개수
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개"
    )

with col2:

    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개"
    )


# ==========================================
# 모델 결과
# ==========================================

st.subheader(
    "📊 모델 성능 비교"
)

st.dataframe(
    result_df.style.format({

        "테스트 평균 오차 (MAE, ℃)": "{:.3f}",

        "2050년 예측값 (℃)": "{:.3f}"

    }),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 가장 성능이 좋은 모델
# ==========================================

mae_values = {

    "1차 (직선)": mae_1,

    "3차 곡선": mae_3,

    "9차 곡선": mae_9
}

best_model = min(
    mae_values,
    key=mae_values.get
)

st.success(
    f"테스트 데이터에서 평균 오차가 가장 작은 모델은 "
    f"**{best_model}**입니다. "
    f"(MAE: {mae_values[best_model]:.3f}℃)"
)


# ==========================================
# 그래프용 연도 만들기
# ==========================================

start_year = int(
    df["연도"].min()
)

graph_years = np.arange(
    start_year,
    2051
)

graph_x = (
    graph_years - year_mean
) / year_std


# 각 모델의 예측 곡선
graph_prediction_1 = model_1(
    graph_x
)

graph_prediction_3 = model_3(
    graph_x
)

graph_prediction_9 = model_9(
    graph_x
)


# ==========================================
# 그래프
# ==========================================

st.subheader(
    "📈 실제 연평균기온과 예측 곡선"
)

fig = go.Figure()


# 실제 기온
fig.add_trace(
    go.Scatter(

        x=df["연도"],

        y=df["평균기온"],

        mode="markers",

        name="실제 연평균기온"
    )
)


# 1차
fig.add_trace(
    go.Scatter(

        x=graph_years,

        y=graph_prediction_1,

        mode="lines",

        name="1차 직선"
    )
)


# 3차
fig.add_trace(
    go.Scatter(

        x=graph_years,

        y=graph_prediction_3,

        mode="lines",

        name="3차 곡선"
    )
)


# 9차
fig.add_trace(
    go.Scatter(

        x=graph_years,

        y=graph_prediction_9,

        mode="lines",

        name="9차 곡선"
    )
)


# 2005년 경계선
fig.add_vline(

    x=2005,

    line_dash="dash",

    annotation_text="2005년: 훈련 → 테스트",

    annotation_position="top"
)


# 2050년 경계선
fig.add_vline(

    x=2050,

    line_dash="dot",

    annotation_text="2050년",

    annotation_position="top"
)


fig.update_layout(

    xaxis_title="연도",

    yaxis_title="평균기온 (℃)",

    hovermode="x unified",

    height=600
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 분석 방법 설명
# ==========================================

with st.expander(
    "📌 분석 방법 보기"
):

    st.write(
        """
        **① 훈련 데이터**

        2005년 이전의 연평균기온만 사용하여
        1차, 3차, 9차 다항회귀 모델을 학습했습니다.

        **② 테스트 데이터**

        2005년부터의 데이터는 모델 학습에
        사용하지 않았습니다.

        학습이 끝난 후 테스트 데이터에 대한
        예측값을 실제 기온과 비교하여 모델을 평가했습니다.

        **③ 테스트 평균 오차(MAE)**

        실제 기온과 예측 기온의 차이를 절댓값으로 만든 뒤
        평균을 계산했습니다.

        따라서 MAE가 작을수록 테스트 데이터에서
        평균적으로 더 정확하게 예측한 모델입니다.

        **④ 2050년 예측**

        훈련 데이터에 포함되지 않은 2050년을
        각각의 모델에 입력하여 미래 기온을 예측했습니다.

        **⑤ 연도 변환**

        1900, 2000과 같은 큰 연도 숫자를 그대로
        고차식 계산에 사용하면 수치적으로 불안정할 수 있습니다.

        그래서 훈련 데이터의 평균과 표준편차를 이용하여
        연도를 작은 값으로 변환한 후 계산했습니다.
        """
    )


# ==========================================
# 원본 데이터 확인
# ==========================================

with st.expander(
    "훈련용 데이터 보기"
):

    st.dataframe(

        train[
            [
                "연도",
                "평균기온",
                "관측일수"
            ]
        ],

        use_container_width=True,

        hide_index=True
    )


with st.expander(
    "테스트용 데이터 보기"
):

    st.dataframe(

        test[
            [
                "연도",
                "평균기온",
                "관측일수"
            ]
        ],

        use_container_width=True,

        hide_index=True
    )
