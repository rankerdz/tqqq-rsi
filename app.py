
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from pandas.tseries.offsets import BDay

st.set_page_config(page_title="ETF RSI 타겟 종가 계산기", page_icon="📈", layout="centered")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #f2f2f3;
    border: 1px solid #e3e3e6;
    padding: 18px;
    border-radius: 16px;
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
    font-size: 1.65rem !important;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 920px;
}
</style>
""", unsafe_allow_html=True)

ETF_OPTIONS = ["QQQ", "TQQQ", "SOXL"]
DEFAULT_TARGETS = [35, 40, 45]

@st.cache_data(show_spinner=False, ttl=3600)
def load_history(symbol):
    df = yf.download(symbol, period="max", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100/(1+rs))

def get_next_trading_day(last_date):
    return pd.Timestamp(last_date) + BDay(1)

def last_rsi_for_next_day_candidate(close, candidate_close, period=14):
    temp = close.copy()
    next_date = get_next_trading_day(temp.index[-1])
    temp.loc[next_date] = candidate_close
    value = calculate_rsi(temp, period).iloc[-1]
    return float(value)

def find_target_close_for_next_day(close, target_rsi, period=14):
    last_close = float(close.iloc[-1])
    low = max(0.01, last_close * 0.1)
    high = last_close * 3
    for _ in range(80):
        mid = (low + high) / 2
        mid_rsi = last_rsi_for_next_day_candidate(close, mid, period)
        if mid_rsi < target_rsi:
            low = mid
        else:
            high = mid
    return round((low+high)/2, 2)

def parse_targets(text):
    targets=[]
    for t in text.split(","):
        try:
            v=float(t.strip())
            if 0<v<100:
                targets.append(v)
        except:
            pass
    if not targets:
        targets=DEFAULT_TARGETS
    return sorted(set(targets))

st.title("ETF RSI 타겟 종가 계산기")
st.caption("위는 다음 거래일 예측 / 아래는 실제 데이터 조회")

with st.sidebar:
    symbol = st.selectbox("종목", ETF_OPTIONS, index=1)
    period = st.number_input("RSI 기간", 2, 50, 14)
    target_text = st.text_input("목표 RSI", "35, 40, 45")

df = load_history(symbol)
close = df["Close"]
rsi = calculate_rsi(close, period)

latest_date = df.index[-1].date()
latest_close = float(close.iloc[-1])
next_date = get_next_trading_day(df.index[-1]).date()

targets = parse_targets(target_text)

c1,c2,c3 = st.columns(3)
c1.metric("예측 기준일", str(latest_date))
c2.metric("예측 대상일", str(next_date))
c3.metric("최신 실제 종가", f"${latest_close:,.2f}")

st.divider()

st.subheader("다음 거래일 목표 RSI 예상 종가")

rows=[]
for t in targets:
    price = find_target_close_for_next_day(close,t,period)
    rows.append({
        "예측 기준일": latest_date,
        "예측 대상일": next_date,
        "목표 RSI": t,
        "예상 종가": f"${price:,.2f}"
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

st.subheader("실제 데이터 조회")

selected_date = st.date_input("날짜 선택", value=latest_date)
ts = pd.Timestamp(selected_date)

c1,c2,c3 = st.columns(3)

if ts not in df.index:
    c1.metric("선택 날짜", selected_date)
    c2.metric("실제 종가", "데이터 없음")
    c3.metric("실제 RSI", "데이터 없음")
else:
    real_close=float(close.loc[ts])
    real_rsi=float(rsi.loc[ts])
    c1.metric("선택 날짜", selected_date)
    c2.metric("실제 종가", f"${real_close:,.2f}")
    c3.metric("실제 RSI", f"{real_rsi:.2f}")
