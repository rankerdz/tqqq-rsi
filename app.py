import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="ETF RSI 타겟 종가 계산기",
    page_icon="📈",
    layout="centered",
)

# 카드(metric) 스타일만 변경
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #f2f2f3;
    border: 1px solid #e3e3e6;
    padding: 20px;
    border-radius: 18px;
}

div[data-testid="stMetric"] * {
    color: #333333 !important;
}

div[data-testid="stMetricLabel"] {
    color: #555555 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #2f2f2f !important;
    font-weight: 700;
    font-size: 1.8rem !important;
}

div[data-testid="stMetricDelta"] {
    color: #444444 !important;
}
</style>
""", unsafe_allow_html=True)


ETF_OPTIONS = ["QQQ", "TQQQ", "SOXL"]
DEFAULT_TARGETS = [35, 40, 45]


@st.cache_data(show_spinner=False, ttl=3600)
def load_history(symbol: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def last_rsi_for_candidate(close: pd.Series, candidate_close: float, period: int = 14) -> float:
    temp = close.copy()
    temp.iloc[-1] = candidate_close
    value = calculate_rsi(temp, period=period).iloc[-1]
    return float(value)


def find_target_close(close: pd.Series, target_rsi: float, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None

    last_close = float(close.iloc[-1])
    low = max(0.01, last_close * 0.1)
    high = last_close * 3.0

    low_rsi = last_rsi_for_candidate(close, low, period)
    high_rsi = last_rsi_for_candidate(close, high, period)

    expand_count = 0
    while target_rsi < low_rsi and expand_count < 20:
        low = max(0.01, low * 0.5)
        low_rsi = last_rsi_for_candidate(close, low, period)
        expand_count += 1

    expand_count = 0
    while target_rsi > high_rsi and expand_count < 20:
        high *= 1.8
        high_rsi = last_rsi_for_candidate(close, high, period)
        expand_count += 1

    if not (low_rsi <= target_rsi <= high_rsi):
        return None

    for _ in range(80):
        mid = (low + high) / 2
        mid_rsi = last_rsi_for_candidate(close, mid, period)

        if mid_rsi < target_rsi:
            low = mid
        else:
            high = mid

    return round((low + high) / 2, 2)


def format_money(value: float | None) -> str:
    if value is None:
        return "데이터 없음"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "데이터 없음"
    return f"${value:,.2f}"


def parse_targets(text: str) -> list[float]:
    targets = []

    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
            if 0 < value < 100:
                targets.append(value)
        except ValueError:
            pass

    if not targets:
        targets = DEFAULT_TARGETS

    return sorted(set(targets))


st.title("ETF RSI 타겟 종가 계산기")
st.caption("실제 일별 종가 기준 · RSI(Wilder 방식) · 개인용 웹앱")

with st.sidebar:
    st.header("설정")
    symbol = st.selectbox("종목", ETF_OPTIONS, index=1)
    period = st.number_input("RSI 기간", min_value=2, max_value=50, value=14, step=1)
    target_text = st.text_input("목표 RSI들", value="35, 40, 45")
    st.caption("쉼표로 구분해서 입력 예: 35, 40, 45")

df = load_history(symbol)

if df.empty:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

close = df["Close"].copy()
rsi = calculate_rsi(close, period=period)

latest_date = df.index[-1].date()
latest_close = float(close.iloc[-1])
latest_rsi = rsi.iloc[-1]
inception_date = df.index[0].date()
targets = parse_targets(target_text)

col1, col2, col3 = st.columns(3)
col1.metric("최신 기준일", str(latest_date))
col2.metric("최신 실제 종가", format_money(latest_close))
col3.metric("최신 RSI", "데이터 없음" if pd.isna(latest_rsi) else f"{float(latest_rsi):.2f}")

st.divider()

st.subheader("목표 RSI별 예상 종가")
st.caption("현재까지 확정된 실제 종가 흐름을 기준으로, 마지막 종가를 가정하여 목표 RSI에 맞는 가격을 계산합니다.")

rows = []
for target in targets:
    target_close = find_target_close(close, target, period=period)
    diff = None if target_close is None else round(target_close - latest_close, 2)
    diff_pct = None if target_close is None else round((target_close / latest_close - 1) * 100, 2)

    rows.append({
        "종목": symbol,
        "목표 RSI": target,
        "예상 종가": "데이터 없음" if target_close is None else f"${target_close:,.2f}",
        "최신 종가 대비": "데이터 없음" if diff is None else f"{diff:+.2f}",
        "최신 종가 대비 %": "데이터 없음" if diff_pct is None else f"{diff_pct:+.2f}%",
    })

target_df = pd.DataFrame(rows)
st.dataframe(target_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("특정 날짜의 실제 RSI 확인")
st.caption(f"{symbol} 상장 이후 전체 일별 종가를 사용합니다. 거래가 없던 날짜면 '데이터 없음'으로 표시합니다.")

selected_date = st.date_input(
    "날짜 선택",
    value=latest_date,
    min_value=inception_date,
    max_value=latest_date,
)

selected_ts = pd.Timestamp(selected_date)
r1, r2, r3 = st.columns(3)

if selected_ts not in df.index:
    r1.metric("선택 날짜", str(selected_date))
    r2.metric("실제 종가", "데이터 없음")
    r3.metric("실제 RSI", "데이터 없음")
    st.info("해당 날짜는 거래 데이터가 없습니다. 주말, 공휴일, 또는 상장 전 날짜일 수 있습니다.")
else:
    selected_close = float(close.loc[selected_ts])
    selected_rsi = rsi.loc[selected_ts]

    r1.metric("선택 날짜", str(selected_date))
    r2.metric("실제 종가", f"${selected_close:,.2f}")
    r3.metric("실제 RSI", "데이터 없음" if pd.isna(selected_rsi) else f"{float(selected_rsi):.2f}")

    hist_df = pd.DataFrame([{
        "Date": str(selected_date),
        "Close": round(selected_close, 2),
        "RSI": None if pd.isna(selected_rsi) else round(float(selected_rsi), 2),
    }])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

st.divider()

with st.expander("설명"):
    st.markdown("""
- **종목 선택**: QQQ / TQQQ / SOXL
- **목표 RSI들**: 기본값은 35, 40, 45
- **예상 종가**: 현재까지 확정된 실제 종가 흐름을 기준으로 계산
- **날짜 조회**: 선택한 날짜의 실제 종가와 실제 RSI 확인
- **데이터 없음 표시**:
  - 거래가 없던 날짜
  - 상장 전 날짜
  - RSI 계산 초기 구간 데이터 부족
""")