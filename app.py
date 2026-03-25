from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="RSI 기반 예상 종가 앱",
    page_icon="📈",
    layout="wide",
)

NY_TZ = ZoneInfo("America/New_York")
KST_TZ = ZoneInfo("Asia/Seoul")


# =========================
# 공통 유틸
# =========================
def now_ny():
    return datetime.now(NY_TZ)


def now_kst():
    return datetime.now(KST_TZ)


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def fmt_date(x):
    if x is None or pd.isna(x):
        return ""
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def parse_tickers(text: str):
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def parse_target_rsis(text: str):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
            if 0 < v < 100:
                values.append(v)
        except Exception:
            pass
    return sorted(set(values))


# =========================
# 미국 거래일 캘린더
# =========================
@st.cache_data(ttl=300)
def get_schedule(start_date, end_date):
    cal = mcal.get_calendar("XNYS")
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    return schedule.copy()


def get_latest_confirmed_session_date(current_ny: datetime):
    """
    미국 정규장 종료 기준으로 최신 '확정된 실제 종가 날짜' 반환
    """
    today = current_ny.date()
    start = today - timedelta(days=20)
    end = today + timedelta(days=7)

    schedule = get_schedule(start, end)
    if schedule.empty:
        return None

    session_dates = [idx.date() for idx in schedule.index]
    today_schedule = schedule.loc[schedule.index.date == today]

    # 오늘이 휴장/주말
    if len(today_schedule) == 0:
        past_sessions = [d for d in session_dates if d < today]
        return max(past_sessions) if past_sessions else None

    market_close = today_schedule.iloc[0]["market_close"].tz_convert(NY_TZ)

    # 오늘 장이 아직 안 끝남
    if current_ny < market_close:
        past_sessions = [d for d in session_dates if d < today]
        return max(past_sessions) if past_sessions else None

    # 오늘 장 종료 후
    return today


def get_next_session_date(ref_date):
    if ref_date is None:
        return None

    start = ref_date
    end = ref_date + timedelta(days=20)

    schedule = get_schedule(start, end)
    if schedule.empty:
        return None

    future_sessions = [idx.date() for idx in schedule.index if idx.date() > ref_date]
    return min(future_sessions) if future_sessions else None


# =========================
# 데이터 로드
# =========================
@st.cache_data(ttl=300)
def load_price_data(ticker: str, period="10y", interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df.index = pd.to_datetime(df.index)

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(NY_TZ).tz_localize(None)

    return df


def prepare_daily_df(df: pd.DataFrame):
    out = df.copy()
    idx = pd.to_datetime(out.index)

    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)

    out.index = idx
    out["_trade_date"] = idx.date
    return out


# =========================
# RSI 계산 (Wilder)
# =========================
def add_rsi_wilder(df: pd.DataFrame, period: int = 14):
    out = df.copy()
    close = out["Close"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(index=out.index, dtype="float64")
    avg_loss = pd.Series(index=out.index, dtype="float64")

    if len(out) <= period:
        out["AvgGain"] = np.nan
        out["AvgLoss"] = np.nan
        out["RSI"] = np.nan
        return out

    first_avg_gain = gain.iloc[1:period + 1].mean()
    first_avg_loss = loss.iloc[1:period + 1].mean()

    avg_gain.iloc[period] = first_avg_gain
    avg_loss.iloc[period] = first_avg_loss

    for i in range(period + 1, len(out)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (period - 1)) + gain.iloc[i]) / period
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (period - 1)) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)

    out["AvgGain"] = avg_gain
    out["AvgLoss"] = avg_loss
    out["RSI"] = rsi

    return out


# =========================
# 최신 실제값
# =========================
def get_latest_actual_row(df: pd.DataFrame, current_ny: datetime):
    latest_confirmed_date = get_latest_confirmed_session_date(current_ny)
    if latest_confirmed_date is None:
        return None

    daily = prepare_daily_df(df)
    eligible = daily[daily["_trade_date"] <= latest_confirmed_date].copy()

    if eligible.empty:
        return None

    return eligible.iloc[-1]


# =========================
# 특정 날짜 실제값 조회
# =========================
def get_row_for_selected_date(df: pd.DataFrame, selected_date: date):
    if selected_date is None:
        return None

    daily = prepare_daily_df(df)
    matched = daily[daily["_trade_date"] == selected_date].copy()

    if matched.empty:
        return None

    return matched.iloc[-1]


# =========================
# 목표 RSI용 다음 거래일 예상 종가
# =========================
def price_for_target_rsi_next_day(latest_close, prev_avg_gain, prev_avg_loss, period, target_rsi):
    latest_close = safe_float(latest_close)
    prev_avg_gain = safe_float(prev_avg_gain)
    prev_avg_loss = safe_float(prev_avg_loss)
    target_rsi = safe_float(target_rsi)

    if None in [latest_close, prev_avg_gain, prev_avg_loss, target_rsi]:
        return None

    if not (0 < target_rsi < 100):
        return None

    n = period
    a = (n - 1) * prev_avg_gain
    b = (n - 1) * prev_avg_loss

    target_rs = target_rsi / (100 - target_rsi)

    # 상승 케이스
    delta_up = target_rs * b - a

    # 하락 케이스
    delta_down = b - (a / target_rs)

    candidates = []

    if delta_up >= 0:
        candidates.append(delta_up)

    if delta_down < 0:
        candidates.append(delta_down)

    if not candidates:
        if abs(delta_up) < 1e-9:
            candidates.append(0.0)
        elif abs(delta_down) < 1e-9:
            candidates.append(0.0)
        else:
            return None

    delta = min(candidates, key=lambda x: abs(x))
    predicted_close = latest_close + delta
    return predicted_close


# =========================
# 화면
# =========================
st.title("📈 RSI 기반 예상 종가 앱")
st.caption("미국 본장 종료 기준으로 최신 실제 종가를 판단하고, 다음 거래일의 목표 RSI별 예상 종가를 계산합니다.")

with st.sidebar:
    st.header("설정")

    ticker_text = st.text_input(
        "종목 입력 (쉼표로 구분)",
        value="QQQ, TQQQ, SOXL, NVDA, TSLA"
    )
    tickers = parse_tickers(ticker_text)

    rsi_period = st.number_input(
        "RSI 기간",
        min_value=2,
        max_value=50,
        value=14,
        step=1
    )

    target_rsi_text = st.text_input(
        "목표 RSI 목록 (쉼표로 구분)",
        value="30, 35, 40, 45, 50, 55, 60, 65, 70"
    )
    target_rsis = parse_target_rsis(target_rsi_text)

    data_period = st.selectbox(
        "데이터 조회 기간",
        options=["2y", "5y", "10y", "max"],
        index=2
    )

    st.markdown("---")
    refresh = st.button("새로고침")

if refresh:
    st.cache_data.clear()

if not tickers:
    st.warning("종목을 하나 이상 입력해줘.")
    st.stop()

if not target_rsis:
    st.warning("목표 RSI를 하나 이상 입력해줘. 예: 30, 35, 40, 50, 60")
    st.stop()

current_ny = now_ny()
current_kst = now_kst()
latest_confirmed_session = get_latest_confirmed_session_date(current_ny)

top1, top2, top3 = st.columns(3)
top1.metric("현재 한국시간", current_kst.strftime("%Y-%m-%d %H:%M:%S"))
top2.metric("현재 뉴욕시간", current_ny.strftime("%Y-%m-%d %H:%M:%S"))
top3.metric(
    "최신 실제 확정 거래일",
    latest_confirmed_session.strftime("%Y-%m-%d") if latest_confirmed_session else "-"
)

st.markdown("---")

# =========================
# 종목 선택
# =========================
selected_ticker = st.selectbox("실제값 확인용 종목 선택", tickers, index=0)

# 데이터 미리 적재
data_map = {}
error_tickers = []

for ticker in tickers:
    try:
        raw = load_price_data(ticker, period=data_period, interval="1d")
        if raw.empty or "Close" not in raw.columns:
            error_tickers.append(ticker)
            continue

        df = add_rsi_wilder(raw, period=int(rsi_period))
        df = df.dropna(subset=["Close"]).copy()

        if df.empty:
            error_tickers.append(ticker)
            continue

        data_map[ticker] = df
    except Exception:
        error_tickers.append(ticker)

if not data_map:
    st.error("데이터를 불러오지 못했어. 종목 코드를 확인해줘.")
    st.stop()

if selected_ticker not in data_map:
    st.warning(f"{selected_ticker} 데이터를 불러오지 못했어.")
    st.stop()

selected_df = data_map[selected_ticker]
selected_daily = prepare_daily_df(selected_df)

min_date = selected_daily["_trade_date"].min()
max_date = selected_daily["_trade_date"].max()

# =========================
# 날짜 선택 실제값
# =========================
st.subheader("선택 날짜 실제 종가 / RSI")

selected_date = st.date_input(
    "날짜 선택",
    value=max_date if isinstance(max_date, date) else date.today(),
    min_value=min_date if isinstance(min_date, date) else None,
    max_value=max_date if isinstance(max_date, date) else None,
)

selected_row = get_row_for_selected_date(selected_df, selected_date)

if selected_row is None:
    actual_selected_df = pd.DataFrame([{
        "종목": selected_ticker,
        "선택 날짜": selected_date,
        "실제 종가": "",
        "실제 RSI": "",
        "비고": "해당 날짜는 거래일 데이터가 없음"
    }])
else:
    actual_selected_df = pd.DataFrame([{
        "종목": selected_ticker,
        "선택 날짜": selected_row["_trade_date"],
        "실제 종가": round(float(selected_row["Close"]), 2),
        "실제 RSI": round(float(selected_row["RSI"]), 2) if pd.notnull(selected_row["RSI"]) else "",
        "비고": ""
    }])

display_actual_selected_df = actual_selected_df.copy()
display_actual_selected_df["선택 날짜"] = display_actual_selected_df["선택 날짜"].apply(fmt_date)
display_actual_selected_df["실제 종가"] = display_actual_selected_df["실제 종가"].apply(
    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
)
display_actual_selected_df["실제 RSI"] = display_actual_selected_df["실제 RSI"].apply(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)

st.dataframe(display_actual_selected_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =========================
# 예측표
# =========================
st.subheader("목표 RSI별 예상 종가")
st.caption("예측 기준일은 미국 본장 종료 기준 최신 실제 종가의 다음 거래일입니다.")

prediction_rows = []

for ticker in tickers:
    if ticker not in data_map:
        continue

    df = data_map[ticker]
    latest_row = get_latest_actual_row(df, current_ny)

    if latest_row is None:
        continue

    latest_date = latest_row["_trade_date"]
    latest_close = safe_float(latest_row["Close"])
    latest_rsi = safe_float(latest_row["RSI"])
    latest_avg_gain = safe_float(latest_row["AvgGain"])
    latest_avg_loss = safe_float(latest_row["AvgLoss"])

    prediction_date = get_next_session_date(latest_date)
    if prediction_date is None:
        prediction_date = latest_date + timedelta(days=1)

    for target_rsi in target_rsis:
        predicted_close = price_for_target_rsi_next_day(
            latest_close=latest_close,
            prev_avg_gain=latest_avg_gain,
            prev_avg_loss=latest_avg_loss,
            period=int(rsi_period),
            target_rsi=target_rsi,
        )

        prediction_rows.append({
            "종목": ticker,
            "예측 기준일": prediction_date,
            "최신 실제 종가일": latest_date,
            "최신 실제 종가": latest_close,
            "최신 실제 RSI": latest_rsi,
            "목표 RSI": target_rsi,
            "예상 종가": predicted_close,
        })

prediction_df = pd.DataFrame(prediction_rows)

if prediction_df.empty:
    st.info("예상 종가 결과가 없습니다.")
else:
    prediction_df = prediction_df.sort_values(["종목", "목표 RSI"]).reset_index(drop=True)

    display_prediction_df = prediction_df.copy()
    display_prediction_df["예측 기준일"] = display_prediction_df["예측 기준일"].apply(fmt_date)
    display_prediction_df["최신 실제 종가일"] = display_prediction_df["최신 실제 종가일"].apply(fmt_date)
    display_prediction_df["최신 실제 종가"] = display_prediction_df["최신 실제 종가"].apply(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
    )
    display_prediction_df["최신 실제 RSI"] = display_prediction_df["최신 실제 RSI"].apply(
        lambda x: f"{x:.2f}" if pd.notnull(x) else ""
    )
    display_prediction_df["목표 RSI"] = display_prediction_df["목표 RSI"].apply(
        lambda x: f"{x:.0f}" if pd.notnull(x) else ""
    )
    display_prediction_df["예상 종가"] = display_prediction_df["예상 종가"].apply(
        lambda x: f"{x:,.2f}" if pd.notnull(x) else ""
    )

    st.dataframe(display_prediction_df, use_container_width=True, hide_index=True)

if error_tickers:
    st.markdown("---")
    st.warning("일부 종목은 불러오지 못했어: " + ", ".join(sorted(set(error_tickers))))

st.markdown("---")
st.caption(
    "주의: 예상 종가는 RSI 수학식 기준의 역산값이며, 실제 시장 예측치가 아니라 "
    "다음 거래일 종가가 특정 RSI가 되기 위해 필요한 이론적 가격입니다."
)